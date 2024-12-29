# training with captions

import importlib
import argparse
import math
import os
from multiprocessing import Value
from typing import List
import toml
import numpy as np
import random
import tools.edm2_loss_mm as edm2_loss_mm
import ast
import matplotlib
import contextlib
matplotlib.use('Agg')  # Set the backend to 'Agg', non-interactive backend
import matplotlib.pyplot as plt
plt.ioff() # Explicitly turn off interactive mode

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device


init_ipex()

from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import deepspeed_utils, sdxl_model_util, strategy_base, strategy_sd, strategy_sdxl

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging
import itertools

import tools.stochastic_accumulator as stochastic_accumulator

logger = logging.getLogger(__name__)

import library.config_util as config_util
import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.sdxl_original_unet import SdxlUNet2DConditionModel


UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23


def get_block_params_to_optimize(unet: SdxlUNet2DConditionModel, block_lrs: List[float]) -> List[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for i, (name, param) in enumerate(unet.named_parameters()):
        if name.startswith("time_embed.") or name.startswith("label_emb."):
            block_index = 0  # 0
        elif name.startswith("input_blocks."):  # 1-9
            block_index = 1 + int(name.split(".")[1])
        elif name.startswith("middle_block."):  # 10-12
            block_index = 10 + int(name.split(".")[1])
        elif name.startswith("output_blocks."):  # 13-21
            block_index = 13 + int(name.split(".")[1])
        elif name.startswith("out."):  # 22
            block_index = 22
        else:
            raise ValueError(f"unexpected parameter name: {name}")

        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize


def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 2:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder1")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            names.append("text_encoder2")

        block_index += 1

    train_util.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)

def determine_grad_sync_context(accelerator, sync_gradients, training_models, lossweightMLP = None):
    if not sync_gradients and accelerator.num_processes > 1:
        if lossweightMLP is not None:
            return accelerator.no_sync(*training_models, lossweightMLP)
        else:
            return accelerator.no_sync(*training_models)
    else:
        return contextlib.nullcontext()

def process_val_batch(batch, tokenize_strategy, text_encoder1, text_encoder2, text_encoding_strategy, unet, vae, noise_scheduler, vae_dtype, weight_dtype, accelerator, args):
    total_loss = 0.0
    timesteps_list = [10, 350, 500, 650, 990]    
    with torch.autograd.grad_mode.inference_mode(mode=True):
        if "latents" in batch and batch["latents"] is not None:
            latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
        else:
            # latentに変換
            latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)

            # NaNが含まれていれば警告を表示し0に置き換える
            if torch.any(torch.isnan(latents)):
                accelerator.print("NaN found in latents, replacing with zeros")
                latents = torch.nan_to_num(latents, 0, out=latents)
        latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            # Text Encoder outputs are cached
            encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoder_outputs_list
            encoder_hidden_states1 = encoder_hidden_states1.to(accelerator.device, dtype=weight_dtype)
            encoder_hidden_states2 = encoder_hidden_states2.to(accelerator.device, dtype=weight_dtype)
            pool2 = pool2.to(accelerator.device, dtype=weight_dtype)
        else:
            input_ids1, input_ids2 = batch["input_ids_list"]
            with accelerator.autocast():
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoding_strategy.encode_tokens(
                    tokenize_strategy, [text_encoder1, text_encoder2], [input_ids1, input_ids2]
                )
                if args.full_fp16:
                    encoder_hidden_states1 = encoder_hidden_states1.to(weight_dtype)
                    encoder_hidden_states2 = encoder_hidden_states2.to(weight_dtype)
                    pool2 = pool2.to(weight_dtype)

            # get size embeddings
            orig_size = batch["original_sizes_hw"]
            crop_size = batch["crop_top_lefts"]
            target_size = batch["target_sizes_hw"]
            embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

            # concat embeddings
            vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
            text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

            # Sample noise
            batch_size = latents.shape[0]
            for fixed_timesteps in timesteps_list:
                with accelerator.autocast():
                    timesteps = torch.full((batch_size,), fixed_timesteps, dtype=torch.long, device=latents.device)
                    
                    noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(
                        args, noise_scheduler, latents, timesteps, False
                    )

                    noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                    if args.v_parameterization:
                        # v-parameterization training
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    loss = train_util.conditional_loss(
                        noise_pred.float(), target.float(), reduction="mean", loss_type="l2", huber_c=huber_c
                    )
                    total_loss += loss

    average_loss = total_loss / len(timesteps_list)    
    return average_loss

def calculate_val_loss(self, 
                        global_step,
                        epoch_step,
                        train_dataloader,
                        val_loss_recorder,
                        val_dataloader,
                        cyclic_val_dataloader,
                        tokenize_strategy, 
                        text_encoder1, 
                        text_encoder2,
                        text_encoding_strategy, 
                        unet, 
                        vae, 
                        noise_scheduler, 
                        vae_dtype, 
                        weight_dtype, 
                        accelerator, 
                        args):
    if global_step != 0 and global_step < args.max_train_steps:
        if val_dataloader is None:
            return None, None, None
        else:
            if args.validation_every_n_step is not None:
                if global_step % int(args.validation_every_n_step) != 0:
                    return None, None, None
            else:
                if epoch_step != len(train_dataloader) - 1:
                    return None, None, None
                
        # Get current seeds from all random number generators
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        torch_cuda_state = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None

        val_Seed = int(args.validation_seed) if args.validation_seed else 23

        random.seed(val_Seed)
        np.random.seed(val_Seed)
        torch.manual_seed(val_Seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(val_Seed)

    accelerator.print("")
    accelerator.print("Validating バリデーション処理...")
    total_loss = 0.0
    with torch.no_grad():
        validation_steps = min(int(args.max_validation_steps), len(val_dataloader)) if args.max_validation_steps is not None else len(val_dataloader)
        val_dataloader_seed = random.randint(global_step, 0x7FFFFFFF)
        val_dataloader_state = random.Random(val_dataloader_seed).getstate()
        for val_step in tqdm(range(validation_steps), desc='Validation Steps'):
            val_original_state = random.getstate()
            random.setstate(val_dataloader_state)
            batch = next(cyclic_val_dataloader)
            val_dataloader_state = random.getstate()
            random.setstate(val_original_state)
            loss = self.process_val_batch(batch, tokenize_strategy, text_encoder1, text_encoder2, text_encoding_strategy, unet, vae, noise_scheduler, vae_dtype, weight_dtype, accelerator, args)
            total_loss += loss.detach().item()
        current_val_loss = total_loss / validation_steps
        val_loss_recorder.add(epoch=0, step=global_step, loss=current_val_loss)   
                    
    average_val_loss: float = val_loss_recorder.moving_average
    logs = {"loss/current_val_loss": current_val_loss, "loss/average_val_loss": average_val_loss}

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if torch_cuda_state and torch.cuda.is_available():
        for i, state in enumerate(torch_cuda_state):
            torch.cuda.set_rng_state(state, i)

    return current_val_loss, average_val_loss, logs

def plot_dynamic_loss_weighting(args, step, model, num_timesteps=1000, device="cpu"):
    """
    Plot the dynamic loss weighting across timesteps using the learned parameters of the DynamicLossModule.

    :param model: The DynamicLossModule instance (after training).
    :param num_timesteps: Total number of timesteps to plot.
    :param scale: Scaling factor for the input time.
    :param device: Device to run computations on.
    """
    with torch.inference_mode():
        # Generate a range of timesteps
        timesteps = torch.linspace(0, num_timesteps - 1, num_timesteps).to(device).int()

        model.train(False)
        loss, loss_scale = model(torch.ones_like(timesteps, device=device), timesteps)
        model.train(True)

        # Plot the dynamic loss weights over time
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps.cpu().numpy(), loss.cpu().numpy(),
                label=f'Dynamic Loss Weight\nStep: {step}')
        plt.xlabel('Timesteps')
        plt.ylabel('Weight')
        plt.title('Dynamic Loss Weighting vs Timesteps')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=1)
        if args.edm2_loss_weighting_generate_graph_y_limit is not None:
            plt.ylim(top=int(args.edm2_loss_weighting_generate_graph_y_limit))
        plt.xlim(left=0, right=num_timesteps)
        plt.xticks(np.arange(0, num_timesteps+1, 100)) 
        # plt.show()

        try:
            os.makedirs(args.edm2_loss_weighting_generate_graph_output_dir, exist_ok=True)
            output_dir = os.path.join(args.edm2_loss_weighting_generate_graph_output_dir, args.output_name)
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"weighting_step_{str(step).zfill(7)}.png"))
        except Exception as e:
            logger.warning(f"Failed to save weighting graph image. Due to: {e}")

        plt.close()

def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    if bool(args.disable_cuda_reduced_precision_operations) if args.disable_cuda_reduced_precision_operations else False:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)

    assert (
        not args.weighted_captions or not args.cache_text_encoder_outputs
    ), "weighted_captions is not supported when caching text encoder outputs / cache_text_encoder_outputsを使うときはweighted_captionsはサポートされていません"
    assert (
        not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

    if args.block_lr:
        block_lrs = [float(lr) for lr in args.block_lr.split(",")]
        assert (
            len(block_lrs) == UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / block_lrは{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値を指定してください"
    else:
        block_lrs = None

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    tokenize_strategy = strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    tokenizers = [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]  # will be removed in the future

    # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
    if args.cache_latents:
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None # placeholder until validation dataset supported for arbitrary

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
        if val_dataset_group is not None:
            assert (
                val_dataset_group.is_latent_cacheable()
            ), "when caching validation latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
    # logit_scale = logit_scale.to(accelerator.device, dtype=weight_dtype)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())
        # assert save_stable_diffusion_format, "save_model_as must be ckpt or safetensors / save_model_asはckptかsafetensorsである必要があります"

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if args.diffusers_xformers:
        # もうU-Netを独自にしたので動かないけどVAEのxformersは動くはず
        accelerator.print("Use xformers by Diffusers")
        # set_diffusers_xformers_flag(unet, True)
        set_diffusers_xformers_flag(vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
        accelerator.print("Disable Diffusers' xformers")
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)
        if val_dataset_group is not None:
            print("Cache validation latents...")
            val_dataset_group.new_cache_latents(vae, accelerator)
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # 学習を準備する：モデルを適切な状態にする
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    train_unet = args.learning_rate != 0
    train_text_encoder1 = False
    train_text_encoder2 = False

    text_encoding_strategy = strategy_sdxl.SdxlTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    if args.train_text_encoder:
        # TODO each option for two text encoders?
        accelerator.print("enable text encoder training")
        if args.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()
        lr_te1 = args.learning_rate_te1 if args.learning_rate_te1 is not None else args.learning_rate  # 0 means not train
        lr_te2 = args.learning_rate_te2 if args.learning_rate_te2 is not None else args.learning_rate  # 0 means not train
        train_text_encoder1 = lr_te1 != 0
        train_text_encoder2 = lr_te2 != 0

        # caching one text encoder output is not supported
        if not train_text_encoder1:
            text_encoder1.to(weight_dtype)
        if not train_text_encoder2:
            text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(train_text_encoder1)
        text_encoder2.requires_grad_(train_text_encoder2)
        text_encoder1.train(train_text_encoder1)
        text_encoder2.train(train_text_encoder2)
    else:
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()

        # TextEncoderの出力をキャッシュする
        if args.cache_text_encoder_outputs:
            # Text Encodes are eval and no grad
            text_encoder_output_caching_strategy = strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, False, is_weighted=args.weighted_captions
            )
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_output_caching_strategy)

            text_encoder1.to(accelerator.device)
            text_encoder2.to(accelerator.device)
            with accelerator.autocast():
                train_dataset_group.new_cache_text_encoder_outputs([text_encoder1, text_encoder2], accelerator.is_main_process)

        accelerator.wait_for_everyone()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    unet.requires_grad_(train_unet)
    if not train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)  # because of unet is not prepared

    training_models = []
    params_to_optimize = []
    if train_unet:
        training_models.append(unet)
        if block_lrs is None:
            params_to_optimize.append({"params": list(unet.parameters()), "lr": args.learning_rate})
        else:
            params_to_optimize.extend(get_block_params_to_optimize(unet, block_lrs))

    if train_text_encoder1:
        training_models.append(text_encoder1)
        params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": args.learning_rate_te1 or args.learning_rate})
    if train_text_encoder2:
        training_models.append(text_encoder2)
        params_to_optimize.append({"params": list(text_encoder2.parameters()), "lr": args.learning_rate_te2 or args.learning_rate})

    # calculate number of trainable parameters
    n_params = 0
    for group in params_to_optimize:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"train unet: {train_unet}, text_encoder1: {train_text_encoder1}, text_encoder2: {train_text_encoder2}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")

    if args.fused_optimizer_groups:
        # fused backward pass: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
        # Instead of creating an optimizer for all parameters as in the tutorial, we create an optimizer for each group of parameters.
        # This balances memory usage and management complexity.

        # calculate total number of parameters
        n_total_params = sum(len(params["params"]) for params in params_to_optimize)
        params_per_group = math.ceil(n_total_params / args.fused_optimizer_groups)

        # split params into groups, keeping the learning rate the same for all params in a group
        # this will increase the number of groups if the learning rate is different for different params (e.g. U-Net and text encoders)
        grouped_params = []
        param_group = []
        param_group_lr = -1
        for group in params_to_optimize:
            lr = group["lr"]
            for p in group["params"]:
                # if the learning rate is different for different params, start a new group
                if lr != param_group_lr:
                    if param_group:
                        grouped_params.append({"params": param_group, "lr": param_group_lr})
                        param_group = []
                    param_group_lr = lr

                param_group.append(p)

                # if the group has enough parameters, start a new group
                if len(param_group) == params_per_group:
                    grouped_params.append({"params": param_group, "lr": param_group_lr})
                    param_group = []
                    param_group_lr = -1

        if param_group:
            grouped_params.append({"params": param_group, "lr": param_group_lr})

        # prepare optimizers for each group
        optimizers = []
        for group in grouped_params:
            _, _, optimizer = train_util.get_optimizer(args, trainable_params=[group])
            optimizers.append(optimizer)
        optimizer = optimizers[0]  # avoid error in the following code

        logger.info(f"using {len(optimizers)} optimizers for fused optimizer groups")

    else:
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

    # prepare dataloader
    # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
    # some strategies can be None
    train_dataset_group.set_current_strategies()

    if val_dataset_group is not None:
        val_dataset_group.set_current_strategies()

    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
        pin_memory=bool(args.pin_data_loader_memory) if args.pin_data_loader_memory else False
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset_group if val_dataset_group is not None else [],
        shuffle=False,
        batch_size=1,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
        pin_memory=bool(args.pin_data_loader_memory) if args.pin_data_loader_memory else False
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
        )

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    if args.fused_optimizer_groups:
        # prepare lr schedulers for each optimizer
        lr_schedulers = [train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes) for optimizer in optimizers]
        lr_scheduler = lr_schedulers[0]  # avoid error in the following code
    else:
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)

    # freeze last layer and final_layer_norm in te1 since we use the output of the penultimate layer
    if train_text_encoder1:
        text_encoder1.text_model.encoder.layers[-1].requires_grad_(False)
        text_encoder1.text_model.final_layer_norm.requires_grad_(False)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            args,
            unet=unet if train_unet else None,
            text_encoder1=text_encoder1 if train_text_encoder1 else None,
            text_encoder2=text_encoder2 if train_text_encoder2 else None,
        )
        # most of ZeRO stage uses optimizer partitioning, so we have to prepare optimizer and ds_model at the same time. # pull/1139#issuecomment-1986790007
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]

    else:
        # acceleratorがなんかよろしくやってくれるらしい
        if train_unet:
            unet = accelerator.prepare(unet)
        if train_text_encoder1:
            text_encoder1 = accelerator.prepare(text_encoder1)
        if train_text_encoder2:
            text_encoder2 = accelerator.prepare(text_encoder2)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    if val_dataset_group is not None:
        val_dataloader = accelerator.prepare(val_dataloader)
        cyclic_val_dataloader = itertools.cycle(val_dataloader)
    else:
        val_dataloader, cyclic_val_dataloader = None, None

    # TextEncoderの出力をキャッシュするときにはCPUへ移動する
    if args.cache_text_encoder_outputs:
        # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
        text_encoder1.to("cpu", dtype=torch.float32)
        text_encoder2.to("cpu", dtype=torch.float32)
        clean_memory_on_device(accelerator.device)
    else:
        # make sure Text Encoders are on GPU
        text_encoder1.to(accelerator.device)
        text_encoder2.to(accelerator.device)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        # During deepseed training, accelerate not handles fp16/bf16|mixed precision directly via scaler. Let deepspeed engine do.
        # -> But we think it's ok to patch accelerator even if deepspeed is enabled.
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    if args.fused_backward_pass:
        # use fused optimizer for backward pass: other optimizers will be supported in the future
        import library.adafactor_fused

        library.adafactor_fused.patch_adafactor_fused(optimizer)
        for param_group in optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:

                    def __grad_hook(tensor: torch.Tensor, param_group=param_group):
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                        optimizer.step_param(tensor, param_group)
                        tensor.grad = None

                    parameter.register_post_accumulate_grad_hook(__grad_hook)

    elif args.fused_optimizer_groups:
        # prepare for additional optimizers and lr schedulers
        for i in range(1, len(optimizers)):
            optimizers[i] = accelerator.prepare(optimizers[i])
            lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

        # counters are used to determine when to step the optimizer
        global optimizer_hooked_count
        global num_parameters_per_group
        global parameter_optimizer_map

        optimizer_hooked_count = {}
        num_parameters_per_group = [0] * len(optimizers)
        parameter_optimizer_map = {}

        for opt_idx, optimizer in enumerate(optimizers):
            for param_group in optimizer.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def optimizer_hook(parameter: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(parameter, args.max_grad_norm)

                            i = parameter_optimizer_map[parameter]
                            optimizer_hooked_count[i] += 1
                            if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                optimizers[i].step()
                                optimizers[i].zero_grad(set_to_none=True)

                        parameter.register_post_accumulate_grad_hook(optimizer_hook)
                        parameter_optimizer_map[parameter] = opt_idx
                        num_parameters_per_group[opt_idx] += 1

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # accelerator.print(
    #     f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    # )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, 
                                          accelerator.device, 
                                          mu=args.laplace_timestep_sampling_mu,
                                          b=args.laplace_timestep_sampling_b)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if args.edm2_loss_weighting:
        values = args.edm2_loss_weighting_optimizer.split(".")
        optimizer_module = importlib.import_module(".".join(values[:-1]))
        case_sensitive_optimizer_type = values[-1]
        opti_args = ast.literal_eval(args.edm2_loss_weighting_optimizer_args)
        opti_lr = float(args.edm2_loss_weighting_optimizer_lr) if args.edm2_loss_weighting_optimizer_lr else 2e-2

        lossweightMLP, MLP_optim = edm2_loss_mm.create_weight_MLP(noise_scheduler,
                                                                    logvar_channels=int(args.edm2_loss_weighting_num_channels) if args.edm2_loss_weighting_num_channels else 128,
                                                                    optimizer=getattr(optimizer_module, case_sensitive_optimizer_type),
                                                                    lr=opti_lr,
                                                                    optimizer_args=opti_args,
                                                                    device=accelerator.device)
        if args.edm2_loss_weighting_initial_weights:
            lossweightMLP.load_weights(args.edm2_loss_weighting_initial_weights)

        if args.edm2_loss_weighting_lr_scheduler:
            def InverseSqrt(
                wrap_optimizer: torch.optim.Optimizer,
                warmup_steps: int = 0,
                constant_steps: int = 0,
                decay_scaling: float = 1.0,
            ):
                def lr_lambda(current_step: int):
                    if current_step <= warmup_steps:
                        return current_step / max(1, warmup_steps)
                    else:
                        return 1 / math.sqrt(max(current_step / max(constant_steps + warmup_steps, 1), 1)**decay_scaling)
                return torch.optim.lr_scheduler.LambdaLR(optimizer=wrap_optimizer, lr_lambda=lr_lambda)
            
            mlp_lr_scheduler = InverseSqrt(
                MLP_optim,
                warmup_steps=args.max_train_steps * float(args.edm2_loss_weighting_lr_scheduler_warmup_percent) if args.edm2_loss_weighting_lr_scheduler_warmup_percent is not None else 0.05,
                constant_steps=args.max_train_steps * float(args.edm2_loss_weighting_lr_scheduler_constant_percent) if args.edm2_loss_weighting_lr_scheduler_constant_percent is not None else 0.15,
                decay_scaling=float(args.edm2_loss_weighting_lr_scheduler_decay_scaling) if args.edm2_loss_weighting_lr_scheduler_decay_scaling is not None else 1.0,
            )
        else:
            mlp_lr_scheduler = train_util.get_dummy_scheduler(MLP_optim)

        mlp_lr_scheduler = accelerator.prepare(mlp_lr_scheduler)
            
        lossweightMLP, MLP_optim = accelerator.prepare(lossweightMLP, MLP_optim)

        if args.edm2_loss_weighting_generate_graph:
            plot_dynamic_loss_weighting(args, 0, lossweightMLP, 1000, accelerator.device)
    else:
        mlp_lr_scheduler = None
        lossweightMLP = None

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "finetuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    # For --sample_at_first
    sdxl_train_util.sample_images(
        accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, [text_encoder1, text_encoder2], unet
    )

    current_val_loss, average_val_loss, val_logs = None, None, None
    if cyclic_val_dataloader is not None:
        current_val_loss, average_val_loss, val_logs = calculate_val_loss(global_step, 0, train_dataloader, val_loss_recorder, val_dataloader, cyclic_val_dataloader, tokenize_strategy, text_encoder1, text_encoder2, text_encoding_strategy, unet, vae, noise_scheduler, vae_dtype, weight_dtype, accelerator, args)
    if len(accelerator.trackers) > 0:
        # log empty object to commit the sample images to wandb
        accelerator.log({}, step=0)

    loss_recorder = train_util.LossRecorder()
    val_loss_recorder = train_util.LossRecorder()

    if args.edm2_loss_weighting:
        loss_scaled_recorder = train_util.LossRecorder()

        if args.sangoi_loss_modifier:
            if args.zero_terminal_snr:
                logger.warning("As zero terminal SNR is set, setting min snr for sangoi loss modifier to zero.")
            if args.min_snr_gamma:
                logger.warning("Min snr gamma and sangoi loss modification both limit the max snr, ignoring min snr gamma in favor of sangoi.")

        if args.stochastic_accumulation:
            if not args.full_bf16:
                logger.warning("""Stochastic accumulation is only applied if using full_bf16. Stochastic accumulation doesn't support fp16, while in mixed precision gradients are fp32.""")

    # Define the number of steps to accumulate gradients
    iter_size = args.gradient_accumulation_steps
    accumulation_counter = 0

    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()
            if args.stochastic_accumulation and args.full_bf16:
                # apply stochastic grad accumulator hooks
                stochastic_accumulator.StochasticAccumulator.assign_hooks(m)

        if args.stochastic_accumulation and args.full_bf16:
            for step, batch in enumerate(train_dataloader):
                current_step.value = global_step

                if args.fused_optimizer_groups:
                    optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}  # reset counter for each step

                # Determine whether we should synchronize gradients
                sync_gradients = (accumulation_counter + 1) % iter_size == 0 or (step + 1 == len(train_dataloader))

                effective_batch_size = accumulation_counter + 1 if sync_gradients else iter_size
                grad_accum_loss_scaling = 1.0 / effective_batch_size

                with determine_grad_sync_context(accelerator, sync_gradients, training_models, lossweightMLP):
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    else:
                        with torch.no_grad():
                            # latentに変換
                            latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)

                            # NaNが含まれていれば警告を表示し0に置き換える
                            if torch.any(torch.isnan(latents)):
                                accelerator.print("NaN found in latents, replacing with zeros")
                                latents = torch.nan_to_num(latents, 0, out=latents)
                    latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                    text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                    if text_encoder_outputs_list is not None:
                        # Text Encoder outputs are cached
                        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoder_outputs_list
                        encoder_hidden_states1 = encoder_hidden_states1.to(accelerator.device, dtype=weight_dtype)
                        encoder_hidden_states2 = encoder_hidden_states2.to(accelerator.device, dtype=weight_dtype)
                        pool2 = pool2.to(accelerator.device, dtype=weight_dtype)
                    else:
                        input_ids1, input_ids2 = batch["input_ids_list"]
                        with torch.set_grad_enabled(args.train_text_encoder):
                            # Get the text embedding for conditioning
                            if args.weighted_captions:
                                input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                                encoder_hidden_states1, encoder_hidden_states2, pool2 = (
                                    text_encoding_strategy.encode_tokens_with_weights(
                                        tokenize_strategy,
                                        [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                        input_ids_list,
                                        weights_list,
                                    )
                                )
                            else:
                                input_ids1 = input_ids1.to(accelerator.device)
                                input_ids2 = input_ids2.to(accelerator.device)
                                encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy,
                                    [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                    [input_ids1, input_ids2],
                                )
                            if args.full_fp16:
                                encoder_hidden_states1 = encoder_hidden_states1.to(weight_dtype)
                                encoder_hidden_states2 = encoder_hidden_states2.to(weight_dtype)
                                pool2 = pool2.to(weight_dtype)

                    # get size embeddings
                    orig_size = batch["original_sizes_hw"]
                    crop_size = batch["crop_top_lefts"]
                    target_size = batch["target_sizes_hw"]
                    embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                    # concat embeddings
                    vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                    text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                    # Sample noise, sample a random timestep for each image, and add noise to the latents,
                    # with noise offset and/or multires noise if specified
                    noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

                    noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

                    # Predict the noise residual
                    with accelerator.autocast():
                        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                    if args.v_parameterization:
                        # v-parameterization training
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                    if (
                        args.min_snr_gamma
                        or args.scale_v_pred_loss_like_noise_pred
                        or args.v_pred_like_loss
                        or args.debiased_estimation_loss
                        or args.masked_loss
                        or args.loss_multipler 
                        or args.loss_multiplier
                        or args.edm2_loss_weighting
                        or args.sangoi_loss_modifier
                    ):
                        # do not mean over batch dimension for snr weight or scale v-pred loss
                        loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c)
                        if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                            loss = apply_masked_loss(loss, batch)
                        loss = loss.mean([1, 2, 3])

                        if args.sangoi_loss_modifier:
                            # Min SNR should be zero for zero_terminal_snr
                            if args.zero_terminal_snr:
                                min_snr = 0
                            else:
                                min_snr = float(args.sangoi_loss_modifier_min_snr)

                            loss = loss * train_util.sangoi_loss_modifier(timesteps, 
                                                                    noise_pred.float(), 
                                                                    target.float(), 
                                                                    noise_scheduler,
                                                                    min_snr,
                                                                    float(args.sangoi_loss_modifier_max_snr))

                        if args.min_snr_gamma and not args.sangoi_loss_modifier:
                            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                        if args.scale_v_pred_loss_like_noise_pred:
                            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                        if args.v_pred_like_loss:
                            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                        if args.debiased_estimation_loss:
                            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)

                        if args.loss_multipler or args.loss_multiplier:
                            loss.mul_(float(args.loss_multipler or args.loss_multiplier) if args.loss_multipler is not None or args.loss_multiplier is not None else 1.0)

                        # For logging
                        pre_scaling_loss = loss.mean()

                        if args.edm2_loss_weighting:
                            loss, loss_scaled = lossweightMLP(loss, timesteps)
                            loss_scaled = loss_scaled.mean()
                            loss_scaled = loss_scaled * grad_accum_loss_scaling

                        loss = loss.mean()  # Mean over batch
                    else:
                        loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "mean", huber_c)
                        pre_scaling_loss = loss

                    # Divide loss by iter_size to average over accumulated steps
                    loss = loss * grad_accum_loss_scaling

                    accelerator.backward(loss)

                    loss = pre_scaling_loss * grad_accum_loss_scaling

                    accumulation_counter += 1

                    if sync_gradients:
                        if not (args.fused_backward_pass or args.fused_optimizer_groups):
                            if args.stochastic_accumulation and args.full_bf16:
                                for m in training_models:
                                    # apply grad buffer back
                                    stochastic_accumulator.StochasticAccumulator.reassign_grad_buffer(m)

                            if args.max_grad_norm != 0.0:
                                params_to_clip = []
                                for m in training_models:
                                    params_to_clip.extend(m.parameters())
                                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                            optimizer.step()

                            if args.edm2_loss_weighting:
                                edm2_loss_weighting_max_grad_norm = float(args.edm2_loss_weighting_max_grad_norm) if args.edm2_loss_weighting_max_grad_norm is not None else 1.0
                                if edm2_loss_weighting_max_grad_norm != 0.0:
                                    params_to_clip = []
                                    params_to_clip.extend(lossweightMLP.parameters())
                                    accelerator.clip_grad_norm_(params_to_clip, edm2_loss_weighting_max_grad_norm).item()

                                MLP_optim.step()

                            lr_scheduler.step()

                            if args.edm2_loss_weighting and mlp_lr_scheduler is not None:
                                mlp_lr_scheduler.step()

                            optimizer.zero_grad(set_to_none=True)

                            if args.edm2_loss_weighting:
                                MLP_optim.zero_grad(set_to_none=True)

                            # Reset accumulation counter
                            accumulation_counter = 0
                        else:
                            if args.edm2_loss_weighting:
                                edm2_loss_weighting_max_grad_norm = float(args.edm2_loss_weighting_max_grad_norm) if args.edm2_loss_weighting_max_grad_norm is not None else 1.0
                                if edm2_loss_weighting_max_grad_norm != 0.0:
                                    params_to_clip = []
                                    params_to_clip.extend(lossweightMLP.parameters())
                                    accelerator.clip_grad_norm_(params_to_clip, edm2_loss_weighting_max_grad_norm).item()

                                MLP_optim.step()

                                if mlp_lr_scheduler is not None:
                                    mlp_lr_scheduler.step()

                                MLP_optim.zero_grad(set_to_none=True)

                            # optimizer.step() and optimizer.zero_grad() are called in the optimizer hook
                            lr_scheduler.step()
                            if args.fused_optimizer_groups:
                                for i in range(1, len(optimizers)):
                                    lr_schedulers[i].step()

                        progress_bar.update(1)
                        global_step += 1

                        if args.edm2_loss_weighting and args.edm2_loss_weighting_generate_graph and (global_step % (int(args.edm2_loss_weighting_generate_graph_every_x_steps) if args.edm2_loss_weighting_generate_graph_every_x_steps else 20) == 0 or global_step >= args.max_train_steps):
                            plot_dynamic_loss_weighting(args, global_step, lossweightMLP, 1000, accelerator.device)

                        sdxl_train_util.sample_images(
                            accelerator,
                            args,
                            None,
                            global_step,
                            accelerator.device,
                            vae,
                            tokenizers,
                            [text_encoder1, text_encoder2],
                            unet,
                        )

                        if cyclic_val_dataloader is not None:
                            current_val_loss, average_val_loss, val_logs = calculate_val_loss(global_step, step, train_dataloader, val_loss_recorder, val_dataloader, cyclic_val_dataloader, tokenize_strategy, text_encoder1, text_encoder2, text_encoding_strategy, unet, vae, noise_scheduler, vae_dtype, weight_dtype, accelerator, args)

                        # 指定ステップごとにモデルを保存
                        if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                                sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                                    args,
                                    False,
                                    accelerator,
                                    src_path,
                                    save_stable_diffusion_format,
                                    use_safetensors,
                                    save_dtype,
                                    epoch,
                                    num_train_epochs,
                                    global_step,
                                    accelerator.unwrap_model(text_encoder1),
                                    accelerator.unwrap_model(text_encoder2),
                                    accelerator.unwrap_model(unet),
                                    vae,
                                    logit_scale,
                                    ckpt_info,
                                )
                                if args.edm2_loss_weighting:
                                    train_util.save_loss_weights_model_on_epoch_end_or_stepwise(args, 
                                                                                            False, 
                                                                                            accelerator.unwrap_model(lossweightMLP),
                                                                                            use_safetensors,
                                                                                            epoch,
                                                                                            num_train_epochs,
                                                                                            global_step)

                current_loss = loss.detach().item() / grad_accum_loss_scaling  # 平均なのでbatch sizeは関係ないはず
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                if args.edm2_loss_weighting:
                    current_loss_scaled = loss_scaled.detach().item() / grad_accum_loss_scaling
                    loss_scaled_recorder.add(epoch=epoch, step=step, loss=current_loss_scaled)
                    average_loss_scaled: float = loss_scaled_recorder.moving_average
                else:
                    current_loss_scaled = None
                    average_loss_scaled = None
                avr_loss: float = loss_recorder.moving_average
                logs = {"loss": current_loss, "avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}

                progress_bar.set_postfix(**logs)

                if val_logs:
                    logs = {**val_logs, **logs}

                if args.edm2_loss_weighting:
                    logs = {"loss/current_loss_scaled": current_val_loss, "loss/average_scaled": average_loss_scaled, **logs}

                if len(accelerator.trackers) > 0:
                    if block_lrs is None:
                        train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                    else:
                        append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)  # U-Net is included in block_lrs

                    if mlp_lr_scheduler is not None:
                        logs[f"lr/edm2"] = mlp_lr_scheduler.get_last_lr()[0]

                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

        else:
            for step, batch in enumerate(train_dataloader):
                current_step.value = global_step

                if args.fused_optimizer_groups:
                    optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}  # reset counter for each step

                with accelerator.accumulate(*training_models, lossweightMLP) if args.edm2_loss_weighting else accelerator.accumulate(*training_models):
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    else:
                        with torch.no_grad():
                            # latentに変換
                            latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)

                            # NaNが含まれていれば警告を表示し0に置き換える
                            if torch.any(torch.isnan(latents)):
                                accelerator.print("NaN found in latents, replacing with zeros")
                                latents = torch.nan_to_num(latents, 0, out=latents)
                    latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                    text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                    if text_encoder_outputs_list is not None:
                        # Text Encoder outputs are cached
                        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoder_outputs_list
                        encoder_hidden_states1 = encoder_hidden_states1.to(accelerator.device, dtype=weight_dtype)
                        encoder_hidden_states2 = encoder_hidden_states2.to(accelerator.device, dtype=weight_dtype)
                        pool2 = pool2.to(accelerator.device, dtype=weight_dtype)
                    else:
                        input_ids1, input_ids2 = batch["input_ids_list"]
                        with torch.set_grad_enabled(args.train_text_encoder):
                            # Get the text embedding for conditioning
                            if args.weighted_captions:
                                input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                                encoder_hidden_states1, encoder_hidden_states2, pool2 = (
                                    text_encoding_strategy.encode_tokens_with_weights(
                                        tokenize_strategy,
                                        [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                        input_ids_list,
                                        weights_list,
                                    )
                                )
                            else:
                                input_ids1 = input_ids1.to(accelerator.device)
                                input_ids2 = input_ids2.to(accelerator.device)
                                encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy,
                                    [text_encoder1, text_encoder2, accelerator.unwrap_model(text_encoder2)],
                                    [input_ids1, input_ids2],
                                )
                            if args.full_fp16:
                                encoder_hidden_states1 = encoder_hidden_states1.to(weight_dtype)
                                encoder_hidden_states2 = encoder_hidden_states2.to(weight_dtype)
                                pool2 = pool2.to(weight_dtype)

                    # get size embeddings
                    orig_size = batch["original_sizes_hw"]
                    crop_size = batch["crop_top_lefts"]
                    target_size = batch["target_sizes_hw"]
                    embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                    # concat embeddings
                    vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                    text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                    # Sample noise, sample a random timestep for each image, and add noise to the latents,
                    # with noise offset and/or multires noise if specified
                    noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

                    noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

                    # Predict the noise residual
                    with accelerator.autocast():
                        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                    if args.v_parameterization:
                        # v-parameterization training
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                    if (
                        args.min_snr_gamma
                        or args.scale_v_pred_loss_like_noise_pred
                        or args.v_pred_like_loss
                        or args.debiased_estimation_loss
                        or args.masked_loss
                        or args.loss_multipler 
                        or args.loss_multiplier
                        or args.edm2_loss_weighting
                        or args.sangoi_loss_modifier
                    ):
                        # do not mean over batch dimension for snr weight or scale v-pred loss
                        loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c)
                        if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                            loss = apply_masked_loss(loss, batch)
                        loss = loss.mean([1, 2, 3])

                        if args.sangoi_loss_modifier:
                            # Min SNR should be zero for zero_terminal_snr
                            if args.zero_terminal_snr:
                                min_snr = 0
                            else:
                                min_snr = float(args.sangoi_loss_modifier_min_snr)

                            loss = loss * train_util.sangoi_loss_modifier(timesteps, 
                                                                    noise_pred.float(), 
                                                                    target.float(), 
                                                                    noise_scheduler,
                                                                    min_snr,
                                                                    float(args.sangoi_loss_modifier_max_snr))

                        if args.min_snr_gamma:
                            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                        if args.scale_v_pred_loss_like_noise_pred:
                            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                        if args.v_pred_like_loss:
                            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                        if args.debiased_estimation_loss:
                            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)

                        if args.loss_multipler or args.loss_multiplier:
                            loss.mul_(float(args.loss_multipler or args.loss_multiplier) if args.loss_multipler is not None or args.loss_multiplier is not None else 1.0)

                        # For logging
                        pre_scaling_loss = loss.mean()

                        if args.edm2_loss_weighting:
                            loss, loss_scaled = lossweightMLP(loss, timesteps)
                            loss_scaled = loss_scaled.mean()

                        loss = loss.mean()  # Mean over batch
                    else:
                        loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "mean", huber_c)
                        pre_scaling_loss = loss

                    accelerator.backward(loss)

                    loss = pre_scaling_loss

                    if not (args.fused_backward_pass or args.fused_optimizer_groups):
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            params_to_clip = []
                            for m in training_models:
                                params_to_clip.extend(m.parameters())
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        optimizer.step()

                        if args.edm2_loss_weighting:
                            edm2_loss_weighting_max_grad_norm = float(args.edm2_loss_weighting_max_grad_norm) if args.edm2_loss_weighting_max_grad_norm is not None else 1.0
                            if accelerator.sync_gradients and edm2_loss_weighting_max_grad_norm != 0.0:
                                params_to_clip = []
                                params_to_clip.extend(lossweightMLP.parameters())
                                accelerator.clip_grad_norm_(params_to_clip, edm2_loss_weighting_max_grad_norm).item()

                            MLP_optim.step()

                        lr_scheduler.step()

                        if args.edm2_loss_weighting and mlp_lr_scheduler is not None:
                            mlp_lr_scheduler.step()

                        optimizer.zero_grad(set_to_none=True)

                        if args.edm2_loss_weighting:
                            MLP_optim.zero_grad(set_to_none=True)
                    else:
                        if args.edm2_loss_weighting:
                            edm2_loss_weighting_max_grad_norm = float(args.edm2_loss_weighting_max_grad_norm) if args.edm2_loss_weighting_max_grad_norm is not None else 1.0
                            if edm2_loss_weighting_max_grad_norm != 0.0:
                                params_to_clip = []
                                params_to_clip.extend(lossweightMLP.parameters())
                                accelerator.clip_grad_norm_(params_to_clip, edm2_loss_weighting_max_grad_norm).item()

                            MLP_optim.step()

                            if mlp_lr_scheduler is not None:
                                mlp_lr_scheduler.step()
                                
                            MLP_optim.zero_grad(set_to_none=True)

                        # optimizer.step() and optimizer.zero_grad() are called in the optimizer hook
                        lr_scheduler.step()
                        if args.fused_optimizer_groups:
                            for i in range(1, len(optimizers)):
                                lr_schedulers[i].step()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if args.edm2_loss_weighting and args.edm2_loss_weighting_generate_graph and (global_step % (int(args.edm2_loss_weighting_generate_graph_every_x_steps) if args.edm2_loss_weighting_generate_graph_every_x_steps else 20) == 0 or global_step >= args.max_train_steps):
                        plot_dynamic_loss_weighting(args, global_step, lossweightMLP, 1000, accelerator.device)

                    sdxl_train_util.sample_images(
                        accelerator,
                        args,
                        None,
                        global_step,
                        accelerator.device,
                        vae,
                        tokenizers,
                        [text_encoder1, text_encoder2],
                        unet,
                    )

                    if cyclic_val_dataloader is not None:
                        current_val_loss, average_val_loss, val_logs = calculate_val_loss(global_step, step, train_dataloader, val_loss_recorder, val_dataloader, cyclic_val_dataloader, tokenize_strategy, text_encoder1, text_encoder2, text_encoding_strategy, unet, vae, noise_scheduler, vae_dtype, weight_dtype, accelerator, args)

                    # 指定ステップごとにモデルを保存
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                            sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                                args,
                                False,
                                accelerator,
                                src_path,
                                save_stable_diffusion_format,
                                use_safetensors,
                                save_dtype,
                                epoch,
                                num_train_epochs,
                                global_step,
                                accelerator.unwrap_model(text_encoder1),
                                accelerator.unwrap_model(text_encoder2),
                                accelerator.unwrap_model(unet),
                                vae,
                                logit_scale,
                                ckpt_info,
                            )
                            if args.edm2_loss_weighting:
                                train_util.save_loss_weights_model_on_epoch_end_or_stepwise(args, 
                                                                                        False, 
                                                                                        accelerator.unwrap_model(lossweightMLP),
                                                                                        use_safetensors,
                                                                                        epoch,
                                                                                        num_train_epochs,
                                                                                        global_step)

                current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                if args.edm2_loss_weighting:
                    current_loss_scaled = loss_scaled.detach().item()
                    loss_scaled_recorder.add(epoch=epoch, step=step, loss=current_loss_scaled)
                    average_loss_scaled: float = loss_scaled_recorder.moving_average
                else:
                    current_loss_scaled = None
                    average_loss_scaled = None
                avr_loss: float = loss_recorder.moving_average
                logs = {"loss": current_loss, "avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}

                progress_bar.set_postfix(**logs)

                if val_logs:
                    logs = {**val_logs, **logs}

                if args.edm2_loss_weighting:
                    logs = {"loss/current_loss_scaled": current_val_loss, "loss/average_scaled": average_loss_scaled, **logs}

                if len(accelerator.trackers) > 0:
                    if block_lrs is None:
                        train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                    else:
                        append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)  # U-Net is included in block_lrs

                    if mlp_lr_scheduler is not None:
                        logs[f"lr/edm2"] = mlp_lr_scheduler.get_last_lr()[0]

                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder1),
                    accelerator.unwrap_model(text_encoder2),
                    accelerator.unwrap_model(unet),
                    vae,
                    logit_scale,
                    ckpt_info,
                )
                if args.edm2_loss_weighting:
                    train_util.save_loss_weights_model_on_epoch_end_or_stepwise(args, 
                                                                            True, 
                                                                            accelerator.unwrap_model(lossweightMLP),
                                                                            use_safetensors,
                                                                            epoch,
                                                                            num_train_epochs,
                                                                            global_step)

        sdxl_train_util.sample_images(
            accelerator,
            args,
            epoch + 1,
            global_step,
            accelerator.device,
            vae,
            tokenizers,
            [text_encoder1, text_encoder2],
            unet,
        )

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.end_training()

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        sdxl_train_util.save_sd_model_on_train_end(
            args,
            src_path,
            save_stable_diffusion_format,
            use_safetensors,
            save_dtype,
            epoch,
            global_step,
            text_encoder1,
            text_encoder2,
            unet,
            vae,
            logit_scale,
            ckpt_info,
        )
        if args.edm2_loss_weighting:
            train_util.save_loss_weights_model_on_train_end(args, use_safetensors, epoch, global_step, accelerator.unwrap_model(lossweightMLP))
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument(
        "--learning_rate_te1",
        type=float,
        default=None,
        help="learning rate for text encoder 1 (ViT-L) / text encoder 1 (ViT-L)の学習率",
    )
    parser.add_argument(
        "--learning_rate_te2",
        type=float,
        default=None,
        help="learning rate for text encoder 2 (BiG-G) / text encoder 2 (BiG-G)の学習率",
    )

    parser.add_argument(
        "--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--block_lr",
        type=str,
        default=None,
        help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
        + f"U-Netの各ブロックの学習率、カンマ区切り、{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    )
    parser.add_argument(
        "--fused_optimizer_groups",
        type=int,
        default=None,
        help="number of optimizers for fused backward pass and optimizer step / fused backward passとoptimizer stepのためのoptimizer数",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=23,
        help="Validation seed"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for validation images out of the training dataset"
    )    
    parser.add_argument(
        "--validation_every_n_step",
        type=int,
        default=None,
        help="Number of train steps for counting validation loss. By default, validation per train epoch is performed"
    )    
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Number of max validation steps for counting validation loss. By default, validation will run entire validation dataset"
    )    

    parser.add_argument(
        "--disable_cuda_reduced_precision_operations",
        action="store_true",
        help="Disables reduced precision for bf16, fp16, and disables use of tf32 to maximize precision at a tiny cost to performance.",
    )

    parser.add_argument(
        "--pin_data_loader_memory",
        action="store_true",
        help="Pins dataloader memory, may speed up dataloader operations.",
    )

    parser.add_argument(
        "--loss_multipler",
        type=float,
        default=None,
        help="A raw multipler to apply to loss.",
    )

    parser.add_argument(
        "--loss_multiplier",
        type=float,
        default=None,
        help="A raw multiplier to apply to loss.",
    )

    parser.add_argument(
        "--edm2_loss_weighting",
        action="store_true",
        help="Use EDM2 loss weighting.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_optimizer",
        type=str,
        default="torch.optim.AdamW",
        help="Fully qualified optimizer class name to use with the edm2 loss weighting optimizer.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_optimizer_lr",
        type=float,
        default=2e-2,
        help="Learning rate as a float for the edm2 loss weighting optimizer.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_optimizer_args",
        type=str,
        default=r"{'weight_decay': 0, 'betas': (0.9,0.99)}",
        help="A JSON object as a string of optimizer args for the edm2 loss weighting optimizer.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_lr_scheduler",
        action="store_true",
        help="Use lr scheduler with EDM2 loss weighting optimizer.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_lr_scheduler_warmup_percent",
        type=float,
        default=0.10,
        help="Percent of training steps to use for warmup.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_lr_scheduler_constant_percent",
        type=float,
        default=0.10,
        help="Percent of training steps to maintain constant LR before decay.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_max_grad_norm",
        type=float,
        default=1.0,
        help="Max grad norm to apply to edm2 loss weighting gradients.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_generate_graph",
        action="store_true",
        help="Enable generation of graph images that show the loss weighting per timestep.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_generate_graph_every_x_steps",
        type=int,
        default=20,
        help="Every x steps generate a graph image.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_generate_graph_output_dir",
        type=str,
        default=None,
        help="""The parent directory where loss weighting graph images should be stored, 
        with sub directories automatically created and named after the model's defined name.""",
    )

    parser.add_argument(
        "--edm2_loss_weighting_generate_graph_y_limit",
        type=int,
        default=None,
        help="""Set the max limit of the y axis, if not set, uses dynamic scaling of the y-axis, which can make it harder to follow. 
        8 is a good value for v-pred + ztsnr without any augmentation (i.e. low min snr gamma, debiased loss, or scaled v-pred loss). 
        If any of the noted augmentations are used, weighting values can reach ~100-150.""",
    )

    parser.add_argument(
        "--edm2_loss_weighting_num_channels",
        type=int,
        default=128,
        help="The number of channels used by for the loss weighting module. Additional channels allows for greater granularity in the weighting.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_initial_weights",
        type=str,
        default=None,
        help="The full filepath to initial weights and state of edm2 weighting model to use instead of random.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_lr_scheduler_decay_scaling",
        type=float,
        default=1.0,
        help="A scaling factor to apply to the decay rate of the edm2_loss_weighting_lr_scheduler, lower values result in slower decay, higher values result in faster decay.",
    )

    parser.add_argument(
            "--immiscible_noise",
            type=int,
            default=None,
            help="Batch size to match noise to latent images. Use Immiscible Noise algorithm to project training images only to nearby noise (from arxiv.org/abs/2406.12303) "
            + "/ ノイズを潜在画像に一致させるためのバッチ サイズ。Immiscible Noise ノイズアルゴリズを使用して、トレーニング画像を近くのノイズにのみ投影します（arxiv.org/abs/2406.12303 より）",
        )
    
    parser.add_argument(
            "--immiscible_diffusion",
            action="store_true",
            help="Use immiscible diffusion to generate noised latents instead of standard noise scheduler. Mutually exclusive with ip noise gamma.",
        )

    parser.add_argument(
            "--sangoi_loss_modifier",
            action="store_true",
            help="Apply sangoi loss modifier to loss.",
        )
    
    parser.add_argument(
            "--sangoi_loss_modifier_min_snr",
            type=float,
            default=1e-4,
            help="Min SNR limit for sangoi loss modifier.",
        )
    
    parser.add_argument(
            "--sangoi_loss_modifier_max_snr",
            type=float,
            default=100,
            help="Max SNR limit for sangoi loss modifier.",
    )

    parser.add_argument(
        "--laplace_timestep_sampling_mu",
        type=float,
        default=None,
        help="Mu parameter for Laplace-based timestep sampling (optional)."
    )
    parser.add_argument(
        "--laplace_timestep_sampling_b",
        type=float,
        default=None,
        help="b parameter for Laplace-based timestep sampling (optional)."
    )

    parser.add_argument(
        "--stochastic_accumulation",
        action="store_true",
        help="Stochastic accumulation"
    )



    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)
