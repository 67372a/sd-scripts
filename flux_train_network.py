import argparse
import copy
import math
import random
from typing import Any, Optional, Union

import torch
from accelerate import Accelerator

from library.device_utils import clean_memory_on_device, init_ipex
from library.ramtorch_util import apply_ramtorch_to_module

init_ipex()

import train_network
from library import (
    flux_models,
    flux_train_utils,
    flux_utils,
    sd3_train_utils,
    strategy_base,
    strategy_flux,
    train_util,
)
from library.custom_train_functions import apply_snr_weight_for_flow_matching, maybe_apply_antithetic_noise_pairing
from library.adaptive_timestep_sampler import AdaptiveTimestepManager
from library import utils
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class FluxNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_schnell: Optional[bool] = None
        self.is_swapping_blocks: bool = False
        self.model_type: Optional[str] = None
        # Flux is a flow-matching model: model_pred is velocity, x0_hat = noisy - sigmas * v.
        # ChromaRadiance `raw` prediction is special-cased in setup_hf_objective below.
        self.hf_prediction_mode = "flow"

    def setup_hf_objective(self, args):
        super().setup_hf_objective(args)
        if getattr(args, "model_type", None) == "chroma_radiance":
            # ChromaRadiance raw: model outputs v = (noisy - x0) / (t + 5e-2); the Tweedie
            # must use the same train-time epsilon as the forward path (spec §2.2).
            self.hf_prediction_mode = "x0_residual_eps"
            self.hf_eps_train = 5e-2

    def get_adaptive_model_type(self, args) -> str:
        return "flow_matching"

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        # sdxl_train_util.verify_sdxl_training_args(args)

        self.model_type = args.model_type  # "flux", "chroma", or "chroma_radiance"
        if self.model_type not in ("chroma", "chroma_radiance"):
            self.use_clip_l = True
        else:
            self.use_clip_l = False  # Chroma / ChromaRadiance do not use CLIP-L
            assert args.apply_t5_attn_mask, "apply_t5_attn_mask must be True for Chroma / ChromaRadiance / Chromaではapply_t5_attn_maskを指定する必要があります"

        if args.fp8_base_unet:
            args.fp8_base = True  # if fp8_base_unet is enabled, fp8_base is also enabled for FLUX.1

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled / cache_text_encoder_outputs_to_diskが有効になっているため、cache_text_encoder_outputsも有効になります"
            )
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used (use --cache_caption_variants for all but token_warmup_step) / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません（token_warmup_step以外は--cache_caption_variantsで対応可能です）"

        # prepare CLIP-L/T5XXL training flags
        self.train_clip_l = not args.network_train_unet_only and self.use_clip_l
        self.train_t5xxl = False  # default is False even if args.network_train_unet_only is False

        if args.max_token_length is not None:
            logger.warning("max_token_length is not used in Flux training / max_token_lengthはFluxのトレーニングでは使用されません")

        args.blocks_to_swap = utils.getattr_cast(args, "blocks_to_swap", 0)

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing / blocks_to_swapはcpu_offload_checkpointingと併用できません"

        # deprecated split_mode option
        if args.split_mode:
            if args.blocks_to_swap is not None:
                logger.warning(
                    "split_mode is deprecated. Because `--blocks_to_swap` is set, `--split_mode` is ignored."
                    " / split_modeは非推奨です。`--blocks_to_swap`が設定されているため、`--split_mode`は無視されます。"
                )
            else:
                logger.warning(
                    "split_mode is deprecated. Please use `--blocks_to_swap` instead. `--blocks_to_swap 18` is automatically set."
                    " / split_modeは非推奨です。代わりに`--blocks_to_swap`を使用してください。`--blocks_to_swap 18`が自動的に設定されました。"
                )
                args.blocks_to_swap = 18  # 18 is safe for most cases

        train_dataset_group.verify_bucket_reso_steps(32)  # TODO check this
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)  # TODO check this

    def load_target_model(self, args, weight_dtype, accelerator):
        # currently offload to cpu for some models

        # if the file is fp8 and we are using fp8_base, we can load it as is (fp8)
        # keep_unet_dtype: always load in native dtype
        if args.keep_unet_dtype:
            loading_dtype = None
        else:
            loading_dtype = None if args.fp8_base else weight_dtype

        # if we load to cpu, flux.to(fp8) takes a long time, so we should load to gpu in future
        _, model = flux_utils.load_flow_model(
            args.pretrained_model_name_or_path,
            loading_dtype,
            "cpu",
            disable_mmap=args.disable_mmap_load_safetensors,
            model_type=self.model_type,
        )
        if args.fp8_base and not args.keep_unet_dtype:
            # check dtype of model
            if model.dtype == torch.float8_e4m3fnuz or model.dtype == torch.float8_e5m2 or model.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
            elif model.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 FLUX model")
            else:
                logger.info(
                    "Cast FLUX model to fp8. This may take a while. You can reduce the time by using fp8 checkpoint."
                    " / FLUXモデルをfp8に変換しています。これには時間がかかる場合があります。fp8チェックポイントを使用することで時間を短縮できます。"
                )
                model.to(torch.float8_e4m3fn)
        elif args.keep_unet_dtype:
            logger.info(f"Keeping FLUX model in its loaded dtype: {model.dtype}")

        if self.model_type == "chroma_radiance" and hasattr(args, "nerf_tile_size") and args.nerf_tile_size > 0:
            model.params.nerf_tile_size = args.nerf_tile_size
            logger.info(f"NeRF head tiling enabled with tile_size={args.nerf_tile_size}")

        # if args.split_mode:
        #     model = self.prepare_split_model(model, weight_dtype, accelerator)

        if args.use_ramtorch:
            logger.info("Applying RamTorch to FLUX model.")
            model = apply_ramtorch_to_module(model, "unet/dit", accelerator.device, model.dtype)

        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        if self.use_clip_l:
            clip_l = flux_utils.load_clip_l(args.clip_l, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        else:
            clip_l = flux_utils.dummy_clip_l()  # dummy CLIP-L for Chroma / ChromaRadiance, which does not use CLIP-L
        clip_l.eval()

        # if the file is fp8 and we are using fp8_base (not unet), we can load it as is (fp8)
        if args.fp8_base and not args.fp8_base_unet:
            loading_dtype = None  # as is
        else:
            loading_dtype = weight_dtype

        # loading t5xxl to cpu takes a long time, so we should load to gpu in future
        t5xxl = flux_utils.load_t5xxl(args.t5xxl, loading_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        t5xxl.eval()
        if args.fp8_base and not args.fp8_base_unet:
            # check dtype of model
            if t5xxl.dtype == torch.float8_e4m3fnuz or t5xxl.dtype == torch.float8_e5m2 or t5xxl.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 t5xxl dtype: {t5xxl.dtype}")
            elif t5xxl.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 T5XXL model")

        if self.model_type == "chroma_radiance":
            # Pixel-space model doesn't need a real VAE
            ae = flux_utils._DummyVAE(weight_dtype, "cpu")
        else:
            ae = flux_utils.load_ae(args.ae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)

        if args.use_ramtorch and not args.cache_text_encoder_outputs:
            clip_l = apply_ramtorch_to_module(clip_l, "clip_l", accelerator.device, weight_dtype)
            t5xxl = apply_ramtorch_to_module(t5xxl, "t5xxl", accelerator.device, t5xxl.dtype)

        if self.model_type == "chroma_radiance":
            model_version = flux_utils.MODEL_VERSION_CHROMA_RADIANCE
        elif self.model_type == "chroma":
            model_version = flux_utils.MODEL_VERSION_CHROMA
        else:
            model_version = flux_utils.MODEL_VERSION_FLUX_V1
        return model_version, [clip_l, t5xxl], ae, model

    def get_tokenize_strategy(self, args):
        # This method is called before `assert_extra_args`, so we cannot use `self.is_schnell` here.
        # Instead, we analyze the checkpoint state to determine if it is schnell.
        if args.model_type not in ("chroma", "chroma_radiance"):
            _, is_schnell, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)
        else:
            is_schnell = False
        self.is_schnell = is_schnell

        if args.t5xxl_max_token_length is None:
            if self.is_schnell:
                t5xxl_max_token_length = 256
            else:
                t5xxl_max_token_length = 512
        else:
            t5xxl_max_token_length = args.t5xxl_max_token_length

        logger.info(f"t5xxl_max_token_length: {t5xxl_max_token_length}")
        return strategy_flux.FluxTokenizeStrategy(t5xxl_max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_flux.FluxTokenizeStrategy):
        return [tokenize_strategy.clip_l, tokenize_strategy.t5xxl]

    def get_latents_caching_strategy(self, args):
        if self.model_type == "chroma_radiance":
            return None  # pixel-space model, no latent caching needed
        latents_caching_strategy = strategy_flux.FluxLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, False,
            cache_dtype=getattr(args, "cache_latents_dtype", "auto"),
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_flux.FluxTextEncodingStrategy(apply_t5_attn_mask=args.apply_t5_attn_mask)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        # check t5xxl is trained or not
        self.train_t5xxl = network.train_t5xxl

        if self.train_t5xxl and args.cache_text_encoder_outputs:
            raise ValueError(
                "T5XXL is trained, so cache_text_encoder_outputs cannot be used / T5XXL学習時はcache_text_encoder_outputsは使用できません"
            )

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            if self.train_clip_l and not self.train_t5xxl:
                return text_encoders[0:1]  # only CLIP-L is needed for encoding because T5XXL is cached
            else:
                return None  # no text encoders are needed for encoding because both are cached
        else:
            return text_encoders  # both CLIP-L and T5XXL are needed for encoding

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [self.train_clip_l, self.train_t5xxl]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            # if the text encoders is trained, we need tokenization, so is_partial is True
            return strategy_flux.FluxTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                is_partial=self.train_clip_l or self.train_t5xxl,
                apply_t5_attn_mask=args.apply_t5_attn_mask,
                cache_dtype=getattr(args, "cache_text_encoder_outputs_dtype", "auto"),
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            logger.info("move text encoders to gpu")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)  # always not fp8
            text_encoders[1].to(accelerator.device)

            if text_encoders[1].dtype == torch.float8_e4m3fn:
                # if we load fp8 weights, the model is already fp8, so we use it as is
                self.prepare_text_encoder_fp8(1, text_encoders[1], text_encoders[1].dtype, weight_dtype)
            else:
                # otherwise, we need to convert it to target dtype
                text_encoders[1].to(weight_dtype)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")

                tokenize_strategy: strategy_flux.FluxTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy: strategy_flux.FluxTextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens_and_masks, args.apply_t5_attn_mask
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # move back to cpu
            if not self.is_train_text_encoder(args):
                logger.info("move CLIP-L back to cpu")
                text_encoders[0].to("cpu")
            logger.info("move t5XXL back to cpu")
            text_encoders[1].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device)

    def sample_images(self, accelerator, args, epoch, global_step, device, ae, tokenizer, text_encoder, flux):
        text_encoders = text_encoder  # for compatibility
        text_encoders = self.get_models_for_text_encoding(args, accelerator, text_encoders)

        flux_train_utils.sample_images(
            accelerator, args, epoch, global_step, flux, ae, text_encoders, self.sample_prompts_te_outputs,
            is_chroma_radiance=(self.model_type == "chroma_radiance"),
        )

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        if self.model_type == "chroma_radiance":
            return images  # pixel-space model, no VAE encoding needed
        return vae.encode(images)

    def shift_scale_latents(self, args, latents):
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        text_encoder_masks,
        unet: flux_models.Flux,
        network,
        weight_dtype,
        train_unet,
        fixed_timesteps=None,
        is_train=True,
    ):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        noise = maybe_apply_antithetic_noise_pairing(args, noise, is_train=is_train)
        bsz = latents.shape[0]

        # get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype, fixed_timesteps=fixed_timesteps, is_train=is_train
        )

        # Store noisy latents for LWD wavelet masking (used in process_batch via base class)
        if is_train and getattr(self, "wavelet_masking_enabled", False):
            self._noisy_latents = noisy_model_input.detach()

        # Store noisy latents for High-Frequency Token loss (4D pre-pack; this is before the
        # chroma pixel-space branch and before pack_latents, so it covers both paths)
        if is_train and self.hf_scale > 0.0:
            self._hf_noisy_latents = noisy_model_input.detach()

        if is_train and self.adv_scale > 0.0:
            self._adv_noisy_latents = noisy_model_input.detach()

        # --- ChromaRadiance: pixel-space forward (no pack/unpack) ---
        if self.model_type == "chroma_radiance":
            l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds
            if not args.apply_t5_attn_mask:
                t5_attn_mask = None

            self.apply_tlora_mask(timesteps)

            with torch.set_grad_enabled(is_train), accelerator.autocast():
                model_pred = unet(
                    x=noisy_model_input,
                    t=timesteps / 1000,
                    txt=t5_out,
                    txt_mask=t5_attn_mask,
                )

            self.clear_tlora_mask_if_needed()

            model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)

            # For chroma_radiance with raw prediction, the model outputs v = (noisy - pred_x0) / (t + 0.05).
            # The target must exactly match this formula using the true x0 (latents) to avoid a 5% noise injection bias.
            if args.model_prediction_type == "raw":
                target = (noisy_model_input - latents) / (timesteps.view(-1, 1, 1, 1) / 1000 + 5e-2)
            else:
                target = noise - latents

            return model_pred, target, timesteps, weighting, noise

        # --- Standard Flux/Chroma latent-space path ---
        # pack latents and get img_ids
        packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

        # get guidance
        # ensure guidance_scale in args is float
        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)

        # get modulation vectors for Chroma
        with accelerator.autocast(), torch.no_grad():
            mod_vectors = unet.get_mod_vectors(timesteps=timesteps / 1000, guidance=guidance_vec, batch_size=bsz)

        if is_train and args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)
            img_ids.requires_grad_(True)
            guidance_vec.requires_grad_(True)
            if mod_vectors is not None:
                mod_vectors.requires_grad_(True)

        # Set T-LoRA timestep mask before the forward pass
        self.apply_tlora_mask(timesteps)

        # Predict the noise residual
        l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds
        if not args.apply_t5_attn_mask:
            t5_attn_mask = None

        def call_dit(img, img_ids, t5_out, txt_ids, l_pooled, timesteps, guidance_vec, t5_attn_mask, mod_vectors):
            # grad is enabled even if unet is not in train mode, because Text Encoder is in train mode
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                model_pred = unet(
                    img=img,
                    img_ids=img_ids,
                    txt=t5_out,
                    txt_ids=txt_ids,
                    y=l_pooled,
                    timesteps=timesteps / 1000,
                    guidance=guidance_vec,
                    txt_attention_mask=t5_attn_mask,
                    mod_vectors=mod_vectors,
                )
            return model_pred

        model_pred = call_dit(
            img=packed_noisy_model_input,
            img_ids=img_ids,
            t5_out=t5_out,
            txt_ids=txt_ids,
            l_pooled=l_pooled,
            timesteps=timesteps,
            guidance_vec=guidance_vec,
            t5_attn_mask=t5_attn_mask,
            mod_vectors=mod_vectors,
        )

        # Clear T-LoRA mask after the forward pass
        self.clear_tlora_mask_if_needed()

        # unpack latents
        model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        # apply model prediction type
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)

        # flow matching loss: this is different from SD3
        target = noise - latents

        # differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                unet.prepare_block_swap_before_forward()
                with torch.no_grad():
                    model_pred_prior = call_dit(
                        img=packed_noisy_model_input[diff_output_pr_indices],
                        img_ids=img_ids[diff_output_pr_indices],
                        t5_out=t5_out[diff_output_pr_indices],
                        txt_ids=txt_ids[diff_output_pr_indices],
                        l_pooled=l_pooled[diff_output_pr_indices],
                        timesteps=timesteps[diff_output_pr_indices],
                        guidance_vec=guidance_vec[diff_output_pr_indices] if guidance_vec is not None else None,
                        t5_attn_mask=t5_attn_mask[diff_output_pr_indices] if t5_attn_mask is not None else None,
                        mod_vectors=mod_vectors[diff_output_pr_indices] if mod_vectors is not None else None,
                    )
                network.set_multiplier(1.0)  # may be overwritten by "network_multipliers" in the next step

                model_pred_prior = flux_utils.unpack_latents(model_pred_prior, packed_latent_height, packed_latent_width)
                model_pred_prior, _ = flux_train_utils.apply_model_prediction_type(
                    args,
                    model_pred_prior,
                    noisy_model_input[diff_output_pr_indices],
                    sigmas[diff_output_pr_indices] if sigmas is not None else None,
                )
                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        return model_pred, target, timesteps, weighting, noise

    def _create_flux_model_fn(self, unet, accelerator, text_conds, text_masks, batch, weight_dtype, args):
        """Create a model_fn for the adaptive timestep sampler that handles the Flux pipeline.

        This model_fn:
        1. Packs the noisy latents
        2. Runs the Flux DiT
        3. Unpacks the output
        4. Applies model_prediction_type preconditioning

        Returns a function(noisy_latents, timesteps, weight_dtype) -> model_pred
        """
        l_pooled, t5_out, txt_ids, t5_attn_mask = text_conds
        if not args.apply_t5_attn_mask:
            t5_attn_mask = None

        def model_fn(noisy_latents, timesteps, wdtype):
            noisy_latents = noisy_latents.to(wdtype)
            bsz = noisy_latents.shape[0]
            packed_latent_height, packed_latent_width = noisy_latents.shape[2] // 2, noisy_latents.shape[3] // 2
            packed_input = flux_utils.pack_latents(noisy_latents)
            img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=noisy_latents.device)
            guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=noisy_latents.device)

            with torch.no_grad(), accelerator.autocast():
                mod_vectors = unet.get_mod_vectors(timesteps=timesteps / 1000, guidance=guidance_vec, batch_size=bsz)
                model_pred = unet(
                    img=packed_input,
                    img_ids=img_ids,
                    txt=t5_out,
                    txt_ids=txt_ids,
                    y=l_pooled,
                    timesteps=timesteps / 1000,
                    guidance=guidance_vec,
                    txt_attention_mask=t5_attn_mask,
                    mod_vectors=mod_vectors,
                )
            model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

            # Apply model prediction type (sigma_scaled, additive, etc.)
            sigmas = timesteps.float().to(noisy_latents.device) / args.num_timesteps if hasattr(args, 'num_timesteps') else timesteps.float().to(noisy_latents.device) / 1000
            sigmas = sigmas.view(-1, 1, 1, 1)
            model_pred, _ = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_latents, sigmas)
            return model_pred

        return model_fn

    def compute_adaptive_delta_before_step(self, unet, noise_scheduler, weight_dtype, accelerator, global_step):
        """Override for Flux: create a Flux-specific model_fn for Algorithm 2."""
        if self.adaptive_manager is None:
            return
        if not self.adaptive_manager.should_update(global_step):
            return

        latents = self._adaptive_last_latents
        noise = self._adaptive_last_noise
        text_conds = self._adaptive_last_text_conds
        adaptive_args = self._adaptive_last_args
        adaptive_batch = self._adaptive_last_batch if self._adaptive_last_batch is not None else {}
        if latents is None or noise is None or text_conds is None or adaptive_args is None:
            return

        text_masks = text_conds[1] if len(text_conds) > 1 else None
        model_fn = self._create_flux_model_fn(unet, accelerator, text_conds, text_masks, adaptive_batch, weight_dtype, adaptive_args)

        # Compute per-timestep losses for a single x_0 at all T timesteps (for the queue)
        self._adaptive_losses_before = self.adaptive_manager.compute_per_timestep_losses(
            latents, noise, model_fn, weight_dtype, label="theta_k pre-step"
        )

        # Cache per-timestep losses for the FULL batch at the current |S| timesteps
        self.adaptive_manager.cache_batch_losses_at_S(
            latents, noise, model_fn, weight_dtype
        )

    def compute_adaptive_delta_after_step(self, unet, noise_scheduler, weight_dtype, accelerator, network):
        """Override for Flux: create a Flux-specific model_fn for Algorithm 2."""
        if self.adaptive_manager is None or self._adaptive_losses_before is None:
            return

        latents = self._adaptive_last_latents
        noise = self._adaptive_last_noise
        text_conds = self._adaptive_last_text_conds
        adaptive_args = self._adaptive_last_args
        adaptive_batch = self._adaptive_last_batch if self._adaptive_last_batch is not None else {}
        if latents is None or noise is None or text_conds is None or adaptive_args is None:
            return

        text_masks = text_conds[1] if len(text_conds) > 1 else None
        model_fn = self._create_flux_model_fn(unet, accelerator, text_conds, text_masks, adaptive_batch, weight_dtype, adaptive_args)

        # Algorithm 2: compute delta approximation
        delta_approx, selected_indices = self.adaptive_manager.compute_delta_approximation(
            model_fn, latents, noise, weight_dtype, self._adaptive_losses_before,
            full_batch_latents=latents, full_batch_noise=noise,
        )

        # Update sampler via policy gradient (Algorithm 1, line 8)
        self.adaptive_manager.update_sampler(delta_approx, latents)

        # Clear cached data
        self._adaptive_losses_before = None
        self._adaptive_last_latents = None
        self._adaptive_last_noise = None
        self._adaptive_last_text_conds = None
        self._adaptive_last_args = None
        self._adaptive_last_batch = None

        # Release the caching allocator's reserved pool after Algorithm 2 sweeps
        # to prevent permanently elevated VRAM.
        if not self._adaptive_disable_empty_cache and accelerator.device.type == "cuda":
            torch.cuda.empty_cache()

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        if args.min_snr_gamma:
            # Convert timesteps (in [0, 1000] range) to flow matching sigmas (in [0, 1] range)
            sigmas = timesteps / noise_scheduler.config.num_train_timesteps
            loss = apply_snr_weight_for_flow_matching(loss, sigmas, args.min_snr_gamma, soft=args.min_snr_gamma_soft)
        return loss

    def get_sai_model_spec(self, args):
        if self.model_type == "chroma_radiance":
            model_description = "chroma_radiance"
        elif self.model_type != "chroma":
            model_description = "schnell" if self.is_schnell else "dev"
        else:
            model_description = "chroma"
        return train_util.get_sai_model_spec(None, args, False, True, False, flux=model_description)

    def update_metadata(self, metadata, args):
        metadata["ss_model_type"] = args.model_type
        metadata["ss_apply_t5_attn_mask"] = args.apply_t5_attn_mask
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_guidance_scale"] = args.guidance_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_model_prediction_type"] = args.model_prediction_type
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

        # Patch Topology Loss config (auxiliary loss runs through the inherited
        # NetworkTrainer.process_batch, so record its config here as well)
        metadata["ss_patch_topology_loss"] = bool(getattr(args, "patch_topology_loss", False))
        metadata["ss_patch_topology_weight"] = getattr(args, "patch_topology_weight", 1.0)
        metadata["ss_patch_topology_tau"] = getattr(args, "patch_topology_tau", 0.1)
        metadata["ss_patch_topology_scale_levels"] = getattr(args, "patch_topology_scale_levels", 2)
        metadata["ss_patch_topology_loss_type"] = getattr(args, "patch_topology_loss_type", "kl")
        metadata["ss_patch_topology_disable_timestep_weight"] = bool(
            getattr(args, "patch_topology_disable_timestep_weight", False)
        )
        metadata["ss_patch_topology_chunk_size"] = getattr(args, "patch_topology_chunk_size", 512)
        metadata["ss_patch_topology_start_step"] = getattr(args, "patch_topology_start_step", 0)
        metadata["ss_patch_topology_warmup_steps"] = getattr(args, "patch_topology_warmup_steps", 0)
        metadata["ss_patch_topology_dynamic_weighting"] = getattr(args, "patch_topology_dynamic_weighting", "none")
        metadata["ss_patch_topology_dwa_temperature"] = getattr(args, "patch_topology_dwa_temperature", 2.0)
        metadata["ss_patch_topology_gradnorm_alpha"] = getattr(args, "patch_topology_gradnorm_alpha", 1.5)
        metadata["ss_patch_topology_dynamic_max_weight"] = getattr(args, "patch_topology_dynamic_max_weight", 10.0)

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        if index == 0:  # CLIP-L
            return super().prepare_text_encoder_grad_ckpt_workaround(index, text_encoder)
        else:  # T5XXL
            text_encoder.encoder.embed_tokens.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        if index == 0:  # CLIP-L
            logger.info(f"prepare CLIP-L for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}")
            text_encoder.to(te_weight_dtype)  # fp8
            text_encoder.text_model.embeddings.to(dtype=weight_dtype)
        else:  # T5XXL

            def prepare_fp8(text_encoder, target_dtype):
                def forward_hook(module):
                    def forward(hidden_states):
                        hidden_gelu = module.act(module.wi_0(hidden_states))
                        hidden_linear = module.wi_1(hidden_states)
                        hidden_states = hidden_gelu * hidden_linear
                        hidden_states = module.dropout(hidden_states)

                        hidden_states = module.wo(hidden_states)
                        return hidden_states

                    return forward

                for module in text_encoder.modules():
                    if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                        # print("set", module.__class__.__name__, "to", target_dtype)
                        module.to(target_dtype)
                    if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.forward = forward_hook(module)

            if flux_utils.get_t5xxl_actual_dtype(text_encoder) == torch.float8_e4m3fn and text_encoder.dtype == weight_dtype:
                logger.info(f"T5XXL already prepared for fp8")
            else:
                logger.info(f"prepare T5XXL for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}, add hooks")
                text_encoder.to(te_weight_dtype)  # fp8
                prepare_fp8(text_encoder, weight_dtype)

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        # if we doesn't swap blocks, we can move the model to device
        flux: flux_models.Flux = unet
        flux = accelerator.prepare(flux, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(flux).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
        accelerator.unwrap_model(flux).prepare_block_swap_before_forward()

        return flux


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)

    parser.add_argument(
        "--split_mode",
        action="store_true",
        # help="[EXPERIMENTAL] use split mode for Flux model, network arg `train_blocks=single` is required"
        # + "/[実験的] Fluxモデルの分割モードを使用する。ネットワーク引数`train_blocks=single`が必要",
        help="[Deprecated] This option is deprecated. Please use `--blocks_to_swap` instead."
        " / このオプションは非推奨です。代わりに`--blocks_to_swap`を使用してください。",
    )
    parser.add_argument(
        "--nerf_tile_size",
        type=int,
        default=0,
        help="tile size for NeRF head processing in Chroma Radiance (0=disabled, 128=process 128 patches at a time) "
        "/ Chroma RadianceのNeRFヘッドのタイルサイズ（0=無効、128=128パッチずつ処理）",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = FluxNetworkTrainer()
    trainer.train(args)
