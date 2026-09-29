import gc
import importlib
import argparse
import math
import os
import typing
from typing import Any, List, Union, Optional
import sys
import random
import time
import json
from multiprocessing import Value
import numpy as np
import ast
import itertools

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.types import Number
from library.device_utils import init_ipex, clean_memory_on_device
from library.edm2_loss_utils import prepare_edm2_loss_weighting, plot_edm2_loss_weighting_check, plot_edm2_loss_weighting
from library.ramtorch_util import apply_ramtorch_to_module

init_ipex()


from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from library import deepspeed_utils, model_util, sai_model_spec, strategy_base, strategy_sd, sai_model_spec
from library import anchor_loss
from library import adv_cache, dist_match
from library import hf_token_loss
from library.strategy_sdxl import SdxlTextEncodingStrategy

import library.train_util as train_util
from library.train_util import DreamBoothDataset
from library.focal_frequency_loss import FocalFrequencyLoss
from library.patch_topology_loss import PatchTopologyLoss, extract_spatial_mask
from library.dynamic_loss_weighting import DynamicLossWeighter, build_weighter_from_args
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.adaptive_timestep_sampler import AdaptiveTimestepManager
from library.custom_train_functions import (
    apply_snr_weight,
    apply_snr_weight_for_flow_matching,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
    _QMCSequenceManager,
)
from library.utils import setup_logging, add_logging_arguments

# T-LoRA timestep-dependent rank masking support
try:
    from lycoris.modules.tlora import set_timestep_mask, clear_timestep_mask, compute_timestep_mask, compute_timestep_mask_batch
    TLORA_AVAILABLE = True
except ImportError:
    TLORA_AVAILABLE = False

# LoRA² adaptive rank regularization support
try:
    from lycoris.modules.lora2 import LoRA2Module
    LORA2_AVAILABLE = True
except ImportError:
    LORA2_AVAILABLE = False

setup_logging()
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchao")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class NetworkTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False
        self.latent_shift = 0.0

        # Weight Noising config (inspired by ai-toolkit-perceptual)
        self.weight_noise_enabled = False

        # T-LoRA timestep-dependent rank masking config
        self.tlora_enabled = False
        self.tlora_max_rank = 0
        self.tlora_min_rank = 1
        self.tlora_mask_alpha = 1.0
        self.tlora_max_timestep = 1000
        self.tlora_use_network_method = False  # True → use network.set_timestep_mask()

        # LoRA² adaptive rank regularization config
        self.lora2_enabled = False
        self.lora2_lambda_r = 1e-4

        # Adaptive Non-uniform Timestep Sampling
        self.adaptive_manager = None
        self._adaptive_last_latents = None
        self._adaptive_last_noise = None
        self._adaptive_last_text_conds = None
        self._adaptive_last_args = None
        self._adaptive_last_batch = None
        self._adaptive_losses_before = None
        self._adaptive_update_pending = False  # True on steps where Algorithm 2 will run
        self._adaptive_disable_empty_cache = False  # --adaptive_sampler_disable_empty_cache

        # Focal Frequency Loss config
        self.ffl_enabled = False
        self.ffl_module = None
        self.ffl_loss_value = None

        # Patch Topology Loss config
        self.patch_topology_enabled = False
        self.patch_topology_loss_module = None
        self.patch_topology_loss_value = None
        self.patch_topology_full_weight = 1.0
        self.patch_topology_start_step = 0
        self.patch_topology_warmup_steps = 0
        self._patch_topology_current_step = 0
        # Dynamic multi-loss weighting (none/dwa/gradnorm); None = static weight
        self.patch_topology_weighter: Optional[DynamicLossWeighter] = None
        self.patch_topology_effective_weight = None  # last effective weight applied (for logging)

        # Latent Wavelet Diffusion (LWD) masking config
        self.wavelet_masking_enabled = False
        self.wavelet_dwt = None
        self._noisy_latents = None  # stored by get_noise_pred_and_target for wavelet map computation
        self._wavelet_mask_ratio = 0.0  # fraction of loss elements masked (for logging)

        # High-Frequency Token latent loss config (see library/hf_token_loss.py)
        self.hf_scale = 0.0              # lambda; 0 = off (bit-identical no-op)
        self.hf_exponent = 1.0           # gamma, must be > 0
        self.hf_patch = 2                # token patch size; must equal model's patchify size
        self.hf_prediction_mode = None   # set by setup_hf_objective / subclasses (None => base derives)
        self.hf_timesteps_in_sigma = False  # True for Anima (timesteps already in [0, 1])
        self.hf_eps_train = 5e-2         # train-time epsilon for x0-residual models (ChromaRadiance raw)
        self.hf_high_noise_snr_cut = (1.0 / 3.0) ** 2  # equivalent to flow sigma_cut=0.75
        self.hf_high_noise_min_weight = 0.10  # retain a small high-noise HF signal
        self.hf_high_noise_power = 1.0       # attenuation exponent above the cutoff
        self._hf_noisy_latents = None    # stored by get_noise_pred_and_target (4D pre-pack)
        self.hf_loss_value = None        # detached scaled HF contribution for logging (tensor)

        # Multiscale MSE x0-prediction ("anchor") loss config (see library/anchor_loss.py)
        self.anchor_scale = 0.0            # term weight; 0 = off (bit-identical no-op)
        self.anchor_levels = 4             # requested Laplacian levels; must be >= 1
        self.anchor_snr_weighting = False  # optional soft SNR(t) gate; inert while scale == 0
        self.anchor_prediction_mode = None  # set by setup_anchor_objective / subclasses (None => base derives)
        self.anchor_timesteps_in_sigma = False  # True for Anima (timesteps already in [0, 1])
        self._anchor_noisy_latents = None  # stored by get_noise_pred_and_target (4D pre-pack)
        self.anchor_loss_value = None      # detached scaled anchor contribution for logging (tensor)
        self._anchor_warned_missing = False  # one-time warning when the term cannot run

        # Stationary multi-scale RFF-MMD distribution matching ("adv") loss.
        self.adv_scale = 0.0
        self.adv_levels = 4
        self.adv_domain = "latent"
        self.adv_features = 256
        self.adv_bandwidths = ""
        self.adv_tokens_per_band = 1024
        self.adv_reference_samples = 4096
        self.adv_rff_seed = 0x5A17C0DE
        self.adv_snr_weighting = False
        self.adv_prediction_mode = None
        self.adv_timesteps_in_sigma = False
        self.adv_references = None
        self.adv_loss_value = None
        self._adv_noisy_latents = None
        self._adv_warned_missing = False

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
        edm2_lr_scheduler=None,
        current_loss_scaled=None,
        average_loss_scaled=None,
        current_loss_edm2=None,
        average_loss_edm2=None,
        current_val_loss=None,
        average_val_loss=None,
        current_ffl_loss=None,
        current_patch_topology_loss=None,
        current_patch_topology_weight=None,
        current_wav_mask_ratio=None,
        current_weight_noise_norm=None,
        current_hf_loss=None,
        current_anchor_loss=None,
        current_adv_loss=None,
        it_s: float = 0.0,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if current_loss_scaled is not None:
            logs["loss/current_scaled"] = current_loss_scaled
            logs["loss/average_scaled"] = average_loss_scaled

        if current_loss_edm2 is not None:
            logs["loss/current_edm2"] = current_loss_edm2
            logs["loss/average_edm2"] = average_loss_edm2

        if current_ffl_loss is not None:
            logs["loss/current_ffl"] = current_ffl_loss

        if current_patch_topology_loss is not None:
            logs["loss/current_patch_topology"] = current_patch_topology_loss

        if current_patch_topology_weight is not None:
            logs["loss/patch_topology_effective_weight"] = current_patch_topology_weight

        if current_wav_mask_ratio is not None:
            logs["loss/wavelet_mask_ratio"] = current_wav_mask_ratio

        if current_weight_noise_norm is not None:
            logs["weight_noise/noise_norm"] = current_weight_noise_norm

        if current_hf_loss is not None:
            logs["loss/current_hf"] = current_hf_loss

        if current_anchor_loss is not None:
            logs["loss/current_anchor"] = current_anchor_loss

        if current_adv_loss is not None:
            logs["loss/current_adv"] = current_adv_loss

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/max_key_norm"] = maximum_norm
        if mean_norm is not None:
            logs["norm/avg_key_norm"] = mean_norm
        if mean_grad_norm is not None:
            logs["norm/avg_grad_norm"] = mean_grad_norm
        if mean_combined_norm is not None:
            logs["norm/avg_combined_norm"] = mean_combined_norm

        if current_val_loss is not None:
            logs["loss/current_val_loss"] = current_val_loss                      
            logs["loss/average_val_loss"] = average_val_loss

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else 1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{i}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower().startswith("Prodigy".lower()):
                opt = lr_scheduler.optimizers[-1] if hasattr(lr_scheduler, "optimizers") else optimizer
                if opt is not None:
                    logs[f"lr/d*lr/{lr_desc}"] = opt.param_groups[i]["d"] * opt.param_groups[i]["lr"]
                    if "effective_lr" in opt.param_groups[i]:
                        logs[f"lr/d*eff_lr/{lr_desc}"] = opt.param_groups[i]["d"] * opt.param_groups[i]["effective_lr"]

            # Log scheduled_lr for Polyak step-size optimizers (e.g. AdamWScheduleFreePlus)
            opt_for_scheduled = lr_scheduler.optimizers[-1] if hasattr(lr_scheduler, "optimizers") else optimizer
            if opt_for_scheduled is not None and "scheduled_lr" in opt_for_scheduled.param_groups[i]:
                logs[f"lr/scheduled/{lr_desc}"] = opt_for_scheduled.param_groups[i]["scheduled_lr"]

        if edm2_lr_scheduler is not None:
            logs[f"lr/edm2"] = edm2_lr_scheduler.get_last_lr()[0]

        if it_s > 0:
            logs["train/it_s"] = round(it_s, 4)

        return logs

    def step_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, global_step, global_step, epoch)

    def epoch_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, epoch, global_step, epoch)

    def accelerator_logging(
        self, accelerator: Accelerator, logs: dict, step_value: int, global_step: int, epoch: int):
        """
        step_value is for tensorboard, other values are for wandb
        """
        tensorboard_tracker = None
        wandb_tracker = None
        other_trackers = []
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tensorboard_tracker = accelerator.get_tracker("tensorboard")
            elif tracker.name == "wandb":
                wandb_tracker = accelerator.get_tracker("wandb")
            else:
                other_trackers.append(accelerator.get_tracker(tracker.name))

        if tensorboard_tracker is not None:
            tensorboard_tracker.log(logs, step=step_value)

        if wandb_tracker is not None:
            logs["global_step"] = global_step
            logs["epoch"] = epoch
            wandb_tracker.log(logs)

        for tracker in other_trackers:
            tracker.log(logs, step=step_value)

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        train_dataset_group.verify_bucket_reso_steps(64)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(64)

    def load_target_model(self, args, weight_dtype, accelerator) -> tuple[str, nn.Module, nn.Module, Optional[nn.Module]]:
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

        if args.use_ramtorch:
            logger.info("Applying RamTorch to SD model.")
            unet = apply_ramtorch_to_module(unet, "unet", accelerator.device, weight_dtype)
            text_encoder = apply_ramtorch_to_module(text_encoder, "clip_l", accelerator.device, weight_dtype)

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, List[nn.Module]]:
        raise NotImplementedError()

    def get_tokenize_strategy(self, args):
        return strategy_sd.SdTokenizeStrategy(args.v2, args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd.SdTokenizeStrategy) -> List[Any]:
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            True, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check,
            cache_dtype=getattr(args, "cache_latents_dtype", "auto"),
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sd.SdTextEncodingStrategy(args.clip_skip)

    def get_text_encoder_outputs_caching_strategy(self, args):
        return None

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        """
        Returns a list of models that will be used for text encoding. SDXL uses wrapped and unwrapped models.
        FLUX.1 and SD3 may cache some outputs of the text encoder, so return the models that will be used for encoding (not cached).
        """
        return text_encoders

    # returns a list of bool values indicating whether each text encoder should be trained
    def get_text_encoders_train_flags(self, args, text_encoders):
        return [True] * len(text_encoders) if self.is_train_text_encoder(args) else [False] * len(text_encoders)

    def get_flow_pixel_counts(self, args, batch, latents):
        return None

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device, dtype=weight_dtype)

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, text_masks, batch, weight_dtype, **kwargs):
        noisy_latents = noisy_latents.to(weight_dtype)
        noise_pred = unet(noisy_latents, timesteps, text_conds[0], text_masks).sample
        return noise_pred

    def all_reduce_network(self, accelerator, network):
        # With a single process there is nothing to synchronize; iterating every
        # parameter here would only add per-step Python overhead (DDP handles
        # the multi-GPU case natively via accelerator.accumulate).
        if accelerator.num_processes <= 1:
            return
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def all_reduce_edm2_model(self, accelerator, edm2_model):
        """Manually synchronize EDM2 model gradients across GPUs."""
        if edm2_model is None:
            return
        if accelerator.num_processes <= 1:
            return
        for param in edm2_model.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def should_sync_ramtorch(self, args, accelerator) -> bool:
        """Whether a full CUDA synchronize is required after backward for RamTorch.

        RamTorch offloads linear weights to CPU; a synchronize is only needed at
        gradient-synchronization boundaries (end of gradient accumulation), not
        after every micro-batch, to avoid serializing CPU/GPU per micro-step.
        """
        return (args.use_ramtorch or args.use_ramtorch_network) and accelerator.sync_gradients

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizers[0], text_encoder, unet)

    # region SD/SDXL

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )

        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        prepare_scheduler_for_custom_training(noise_scheduler, device)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae: AutoencoderKL, images: torch.FloatTensor) -> torch.FloatTensor:
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, args, latents: torch.FloatTensor) -> torch.FloatTensor:
        return (latents - self.latent_shift) * self.vae_scale_factor

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        text_encoder_masks,
        unet,
        network,
        weight_dtype,
        train_unet,
        fixed_timesteps=None,
        is_train=True,
    ):
        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        encoder_attention_mask_bias = text_encoder_masks[1] #[(1 - t.to(dtype=text_encoder_conds[0].dtype)).unsqueeze(1) * -10000.0 for t in text_encoder_masks]

        pixel_counts = None
        if hasattr(self, "get_flow_pixel_counts"):
            pixel_counts = self.get_flow_pixel_counts(args, batch, latents.device)

        # Adaptive timestep sampling: use Beta distribution sampler if enabled
        adaptive_fixed_timesteps = fixed_timesteps
        if is_train and self.adaptive_manager is not None and fixed_timesteps is None:
            adaptive_fixed_timesteps = self.adaptive_manager.sample_timesteps(
                latents, noise_scheduler.config.num_train_timesteps
            )
            # Store latents and args for Algorithm 2 (delta computation after optimizer step).
            # Only pin tensors when this is an update step to avoid wasting VRAM on non-update steps.
            if self._adaptive_update_pending:
                self._adaptive_last_latents = latents.detach()
                self._adaptive_last_args = args

        noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(
            args, noise_scheduler, latents, fixed_timesteps=adaptive_fixed_timesteps, is_train=is_train, pixel_counts=pixel_counts
        )

        # Now that noise is actually computed, store it for Algorithm 2
        if is_train and self.adaptive_manager is not None and fixed_timesteps is None and self._adaptive_update_pending:
            self._adaptive_last_noise = noise.detach()

        # Store noisy latents for LWD wavelet masking (used in process_batch)
        if is_train and getattr(self, "wavelet_masking_enabled", False):
            self._noisy_latents = noisy_latents.detach() if isinstance(noisy_latents, torch.Tensor) else None

        # Store noisy latents for High-Frequency Token loss (must be 4D pre-pack;
        # for inpainting this is the 4-channel latent, before the mask concat below).
        if is_train and self.hf_scale > 0.0:
            self._hf_noisy_latents = noisy_latents.detach() if isinstance(noisy_latents, torch.Tensor) else None

        # Store noisy latents for the multiscale x0-prediction anchor loss (must be 4D
        # pre-pack; for inpainting this is the 4-channel latent, before the mask concat).
        if is_train and self.anchor_scale > 0.0:
            self._anchor_noisy_latents = noisy_latents.detach() if isinstance(noisy_latents, torch.Tensor) else None

        if is_train and self.adv_scale > 0.0:
            self._adv_noisy_latents = noisy_latents.detach() if isinstance(noisy_latents, torch.Tensor) else None

        # ensure the hidden state will require grad
        if is_train and args.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)

        # Set T-LoRA timestep mask before the forward pass.
        # Path 1 (full T-LoRA, algo="tlora"): mask is integral to the
        #   architecture — used at inference time (ComfyUI loader applies
        #   per-step masking).  Always apply, including validation.
        # Path 2 (LoCon flag, use_timestep_mask=True): mask is a training
        #   curriculum technique — checkpoint saves as standard lora_up/lora_down
        #   with all ranks active at inference.  Only apply during training.
        if is_train or not self.tlora_use_network_method:
            self.apply_tlora_mask(timesteps)
        
        # For inpainting models: concatenate [noisy_latents, mask, masked_latents] -> 9-channel UNet input
        unet_latents = noisy_latents
        if batch.get("masked_latents") is not None:
            mask = torch.nn.functional.interpolate(
                batch["masks"].to(weight_dtype), size=noisy_latents.shape[2:]
            )
            unet_latents = torch.cat([noisy_latents, mask, batch["masked_latents"].to(weight_dtype)], dim=1)

        # Predict the noise residual
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            noise_pred = self.call_unet(
                args,
                accelerator,
                unet,
                unet_latents.requires_grad_(train_unet),
                timesteps,
                text_encoder_conds,
                encoder_attention_mask_bias,
                batch,
                weight_dtype,
            )

        # Clear T-LoRA mask after the forward pass (matching the guard above)
        if is_train or not self.tlora_use_network_method:
            self.clear_tlora_mask_if_needed()

        # Upcast for grokking
        latents = latents.to(torch.float64)
        noise = noise.to(torch.float64)

        if getattr(args, "flow_model", False):
            target = noise - latents
        elif args.v_parameterization:
            # v-parameterization training
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        # differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    noise_pred_prior = self.call_unet(
                        args,
                        accelerator,
                        unet,
                        noisy_latents,
                        timesteps,
                        text_encoder_conds,
                        encoder_attention_mask_bias,
                        batch,
                        weight_dtype,
                        indices=diff_output_pr_indices,
                    )
                network.set_multiplier(1.0)  # may be overwritten by "network_multipliers" in the next step
                target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

        return noise_pred, target, timesteps, None, noise

    def post_process_loss(self, loss, args, timesteps: torch.IntTensor, noise_scheduler) -> torch.FloatTensor:
        if getattr(args, "flow_model", False):
            # For flow-matching models (enabled via --flow_model), apply flow-aware
            # Min-SNR-gamma instead of the DDPM-style apply_snr_weight (which
            # requires a DDPM scheduler with alphas_cumprod).
            if args.min_snr_gamma:
                sigmas = timesteps / noise_scheduler.config.num_train_timesteps
                loss = apply_snr_weight_for_flow_matching(loss, sigmas, args.min_snr_gamma, soft=args.min_snr_gamma_soft)
        else:
            if args.min_snr_gamma:
                loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization, soft=args.min_snr_gamma_soft)
            if args.scale_v_pred_loss_like_noise_pred:
                loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
            if args.v_pred_like_loss:
                loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
            if args.debiased_estimation_loss:
                loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)
        return loss

    def setup_hf_objective(self, args):
        """Resolve the High-Frequency Token latent loss config from args.

        Base objective resolution: eps/v-pred/flow (DDPM-family). Subclasses set
        `self.hf_prediction_mode` (and optionally `hf_timesteps_in_sigma`) in their
        `__init__` or override this method; a non-None mode is never clobbered here.
        """
        self.hf_scale = float(getattr(args, "hf_scale", 0.0) or 0.0)
        self.hf_exponent = float(getattr(args, "hf_exponent", 1.0) or 1.0)
        self.hf_patch = int(getattr(args, "hf_patch", 2) or 2)
        self.hf_high_noise_min_weight = float(getattr(args, "hf_high_noise_min_weight", 0.10))
        self.hf_high_noise_power = float(getattr(args, "hf_high_noise_power", 1.0))
        self.hf_high_noise_snr_cut = float(getattr(args, "hf_high_noise_snr_cut", 1.0 / 9.0))
        hf_token_loss.validate_hf_args(self.hf_scale, self.hf_exponent, self.hf_patch)
        hf_token_loss.validate_hf_high_noise_gate_args(
            self.hf_high_noise_min_weight,
            self.hf_high_noise_power,
            self.hf_high_noise_snr_cut,
        )
        if self.hf_prediction_mode is None:
            if getattr(args, "flow_model", False):
                self.hf_prediction_mode = "flow"
            elif args.v_parameterization:
                self.hf_prediction_mode = "vpred_ddpm"
            else:
                self.hf_prediction_mode = "eps_ddpm"
        if self.hf_scale > 0.0:
            logger.info(
                f"High-Frequency Token latent loss enabled: scale={self.hf_scale}, "
                f"exponent={self.hf_exponent}, patch={self.hf_patch}, mode={self.hf_prediction_mode}, "
                f"high_noise_snr_cut={self.hf_high_noise_snr_cut}, "
                f"high_noise_min_weight={self.hf_high_noise_min_weight}, "
                f"high_noise_power={self.hf_high_noise_power}"
            )

    def setup_anchor_objective(self, args):
        """Resolve the multiscale MSE x0-prediction ("anchor") loss config from args.

        Base objective resolution: eps/v-pred/flow (DDPM-family). Subclasses set
        `self.anchor_prediction_mode` (and optionally `anchor_timesteps_in_sigma`) in
        their `__init__`; a non-None mode is never clobbered here.
        """
        self.anchor_scale = float(getattr(args, "anchor_scale", 0.0) or 0.0)
        anchor_levels_arg = getattr(args, "anchor_levels", 4)
        self.anchor_levels = 4 if anchor_levels_arg is None else int(anchor_levels_arg)
        self.anchor_snr_weighting = bool(getattr(args, "anchor_snr_weighting", False))
        anchor_loss.validate_anchor_args(self.anchor_scale, self.anchor_levels)
        if self.anchor_prediction_mode is None:
            if getattr(args, "flow_model", False):
                self.anchor_prediction_mode = "flow"
            elif args.v_parameterization:
                self.anchor_prediction_mode = "vpred_ddpm"
            else:
                self.anchor_prediction_mode = "eps_ddpm"
        if self.anchor_scale > 0.0:
            logger.info(
                f"Multiscale MSE x0-prediction anchor loss enabled: scale={self.anchor_scale}, "
                f"levels={self.anchor_levels}, snr_weighting={self.anchor_snr_weighting}, "
                f"mode={self.anchor_prediction_mode}"
            )

    def setup_adv_objective(self, args):
        """Resolve and validate the stationary multi-scale RFF-MMD objective."""
        self.adv_scale = float(getattr(args, "adv_scale", 0.0) or 0.0)
        levels = getattr(args, "adv_levels", 4)
        self.adv_levels = 4 if levels is None else int(levels)
        self.adv_domain = str(getattr(args, "adv_domain", "latent") or "latent")
        self.adv_features = int(getattr(args, "adv_features", 256))
        self.adv_bandwidths = str(getattr(args, "adv_bandwidths", "") or "")
        self.adv_tokens_per_band = int(getattr(args, "adv_tokens_per_band", 1024))
        self.adv_reference_samples = int(getattr(args, "adv_reference_samples", 4096))
        self.adv_rff_seed = int(getattr(args, "adv_rff_seed", 0x5A17C0DE))
        self.adv_snr_weighting = bool(getattr(args, "adv_snr_weighting", False))
        dist_match.validate_adv_args(
            self.adv_scale,
            self.adv_levels,
            self.adv_features,
            self.adv_tokens_per_band,
            self.adv_reference_samples,
            self.adv_bandwidths,
        )
        if self.adv_domain not in ("latent", "tae_pixels"):
            raise ValueError("adv_domain must be 'latent' or 'tae_pixels'")
        if self.adv_scale > 0.0 and self.adv_domain != "latent":
            raise ValueError(
                "adv_domain='tae_pixels' is not supported by this train_network trainer; "
                "use adv_domain='latent'"
            )
        if self.adv_scale > 0.0 and (
            getattr(self, "model_type", None) == "chroma_radiance"
            or getattr(args, "model_type", None) == "chroma_radiance"
        ):
            raise ValueError(
                "stationary adv matching for ChromaRadiance requires the pixel/TAE reference path, "
                "which is not available in this train_network implementation"
            )

        if self.adv_prediction_mode is None:
            self.adv_prediction_mode = (
                self.hf_prediction_mode or self.anchor_prediction_mode
            )
        if self.adv_prediction_mode is None:
            if getattr(args, "flow_model", False):
                self.adv_prediction_mode = "flow"
            elif args.v_parameterization:
                self.adv_prediction_mode = "vpred_ddpm"
            else:
                self.adv_prediction_mode = "eps_ddpm"
        self.adv_timesteps_in_sigma = bool(
            self.hf_timesteps_in_sigma or self.anchor_timesteps_in_sigma
        )
        if self.adv_scale > 0.0:
            logger.info(
                f"Stationary multi-scale RFF-MMD loss enabled: scale={self.adv_scale}, "
                f"levels={self.adv_levels}, features={self.adv_features}, "
                f"tokens_per_band={self.adv_tokens_per_band}, domain={self.adv_domain}, "
                f"mode={self.adv_prediction_mode}, snr_weighting={self.adv_snr_weighting}"
            )

    def _adv_dataset_signature(self, train_dataset_group, args) -> str:
        """Fingerprint canonical training sources and every cache-affecting option."""
        records = []
        datasets = getattr(train_dataset_group, "datasets", [train_dataset_group])
        for dataset in datasets:
            source_dataset = getattr(dataset, "dreambooth_dataset_delegate", dataset)
            image_data = getattr(source_dataset, "image_data", {})
            image_to_subset = getattr(source_dataset, "image_to_subset", {})
            for image_key, info in image_data.items():
                subset = image_to_subset.get(image_key)
                if bool(getattr(info, "is_val", False)) or bool(getattr(info, "is_reg", False)):
                    continue
                if subset is not None and (bool(getattr(subset, "is_val", False)) or bool(getattr(subset, "is_reg", False))):
                    continue
                source_path = getattr(info, "latents_npz", None) or getattr(info, "absolute_path", None)
                stat = None
                if source_path and os.path.isfile(source_path):
                    file_stat = os.stat(source_path)
                    stat = (file_stat.st_size, file_stat.st_mtime_ns)
                records.append(
                    (
                        str(image_key),
                        str(source_path),
                        stat,
                        tuple(getattr(info, "bucket_reso", ()) or ()),
                        tuple(
                            sorted(
                                (int(k), tuple(v[0]))
                                for k, v in (getattr(info, "jitter_bucket_info", {}) or {}).items()
                            )
                        ),
                        (
                            bool(getattr(subset, "flip_aug", False)),
                            bool(getattr(subset, "color_aug", False)),
                            bool(getattr(subset, "gamma_aug", False)),
                            bool(getattr(subset, "random_crop", False)),
                            int(getattr(subset, "num_repeats", 1) or 1),
                        ),
                    )
                )
        records.sort(key=lambda row: (row[0], row[1]))
        payload = {
            "cache_version": adv_cache.CACHE_VERSION,
            "sources": records,
            "pretrained_model": getattr(args, "pretrained_model_name_or_path", None),
            "vae": getattr(args, "vae", None),
            "vae_custom_scale": getattr(args, "vae_custom_scale", None),
            "vae_custom_shift": getattr(args, "vae_custom_shift", None),
            "seed": getattr(args, "seed", None),
            "cache_latents_dtype": getattr(args, "cache_latents_dtype", "auto"),
            "cache_aug_variants": getattr(args, "cache_aug_variants", 0),
            "cache_aug_refresh_epochs": getattr(args, "cache_aug_refresh_epochs", 0),
            "latent_scale_factor": self.vae_scale_factor,
            "latent_shift_factor": self.latent_shift,
            "levels": self.adv_levels,
            "domain": self.adv_domain,
            "features": self.adv_features,
            "bandwidths": self.adv_bandwidths,
            "tokens_per_band": self.adv_tokens_per_band,
            "reference_samples": self.adv_reference_samples,
            "rff_seed": self.adv_rff_seed,
        }
        return adv_cache.fingerprint_payload(payload)

    @staticmethod
    def _adv_cache_sources(train_dataset_group):
        """Yield lazy loaders for canonical training latents (plus cached variants)."""
        sources = []

        def add_source(loader, grid_key=None, source_weight=1):
            if grid_key is not None:
                loader.adv_grid_key = grid_key
            loader.adv_weight = max(1, int(source_weight))
            sources.append(loader)

        def latent_grid_key(latent):
            shape = tuple(latent.shape)
            if len(shape) == 5:
                return f"{shape[-2]}x{shape[-1]}"
            if len(shape) == 4:
                return f"{shape[-2]}x{shape[-1]}"
            if len(shape) == 3:
                return f"{shape[-2]}x{shape[-1]}"
            return None

        def add_tensor(latent, source_weight=1):
            add_source(lambda latent=latent: latent, latent_grid_key(latent), source_weight)

        datasets = getattr(train_dataset_group, "datasets", [train_dataset_group])
        seen_info = set()
        for dataset in datasets:
            source_dataset = getattr(dataset, "dreambooth_dataset_delegate", dataset)
            image_data = getattr(source_dataset, "image_data", {})
            image_to_subset = getattr(source_dataset, "image_to_subset", {})
            latent_strategy = getattr(source_dataset, "latents_caching_strategy", None)
            if latent_strategy is None:
                try:
                    from library.strategy_base import LatentsCachingStrategy

                    latent_strategy = LatentsCachingStrategy.get_strategy()
                except (AttributeError, RuntimeError):
                    latent_strategy = None

            for image_key, info in image_data.items():
                if id(info) in seen_info:
                    continue
                seen_info.add(id(info))
                subset = image_to_subset.get(image_key)
                if bool(getattr(info, "is_val", False)) or bool(getattr(info, "is_reg", False)):
                    continue
                if subset is not None and (bool(getattr(subset, "is_val", False)) or bool(getattr(subset, "is_reg", False))):
                    continue
                source_weight = max(1, int(getattr(subset, "num_repeats", 1) or 1))

                # Resolution-jitter latent caches are separate clean grids.
                jitter_latents = getattr(info, "latents_by_reso", {}) or {}
                if jitter_latents:
                    for cached in jitter_latents.values():
                        latent = cached[0]
                        if latent is not None:
                            add_tensor(latent, source_weight)
                        flipped = cached[1]
                        if bool(getattr(subset, "flip_aug", False)) and flipped is not None:
                            add_tensor(flipped, source_weight)
                    continue

                latent = getattr(info, "latents", None)
                variants = getattr(info, "latents_aug_variants", None) or []
                if latent is not None:
                    add_tensor(latent, source_weight)
                    if bool(getattr(subset, "flip_aug", False)) and getattr(info, "latents_flipped", None) is not None:
                        flipped = info.latents_flipped
                        add_tensor(flipped, source_weight)
                    for variant in variants:
                        variant_latent = variant.get("latents") if isinstance(variant, dict) else None
                        if variant_latent is not None:
                            add_tensor(variant_latent, source_weight)
                    continue

                latent_path = getattr(info, "latents_npz", None)
                if latent_path is None or latent_strategy is None:
                    continue
                resolutions = [tuple(info.bucket_reso)] if getattr(info, "bucket_reso", None) is not None else []
                jitter_info = getattr(info, "jitter_bucket_info", {}) or {}
                if jitter_info:
                    resolutions = list(dict.fromkeys(tuple(value[0]) for value in jitter_info.values()))
                if not resolutions:
                    continue
                try:
                    variant_count = source_dataset.get_effective_aug_variant_count(subset) if subset is not None else 1
                except (AttributeError, TypeError):
                    variant_count = 1
                for resolution in resolutions:
                    grid_key = f"{int(resolution[1]) // 8}x{int(resolution[0]) // 8}"
                    for variant_index in range(max(1, int(variant_count))):
                        def load_variant(path=latent_path, reso=resolution, vi=variant_index, strategy=latent_strategy):
                            return strategy.load_latents_from_disk(path, reso, variant=vi)[0]

                        add_source(load_variant, grid_key, source_weight)
                    if bool(getattr(subset, "flip_aug", False)):
                        def load_flipped(path=latent_path, reso=resolution, strategy=latent_strategy):
                            flipped_latents = strategy.load_latents_from_disk(path, reso, variant=0)[3]
                            if flipped_latents is None:
                                raise ValueError(f"flip augmentation requested but no flipped latent is cached in {path}")
                            return flipped_latents

                        add_source(load_flipped, grid_key, source_weight)
        return sources

    def prepare_adv_reference(self, args, train_dataset_group, accelerator):
        """Load or build the fixed dataset embedding before training begins."""
        if self.adv_scale <= 0.0:
            return
        signature = self._adv_dataset_signature(train_dataset_group, args)
        output_dir = getattr(args, "output_dir", None) or "."
        cache_dir = getattr(args, "adv_cache_dir", None) or os.path.join(output_dir, "adv_cache")
        cache_path = os.path.join(cache_dir, f"reference-{signature[:20]}.pt")
        if accelerator.is_main_process:
            try:
                adv_cache.load_adv_cache(cache_path, signature, accelerator.device)
                logger.info(f"Loaded stationary adv reference cache: {cache_path}")
            except (OSError, ValueError, RuntimeError, EOFError) as exc:
                if os.path.exists(cache_path):
                    logger.warning(f"Rebuilding invalid stationary adv reference cache ({exc})")
                sources = self._adv_cache_sources(train_dataset_group)
                transformed_sources = []
                for source in sources:
                    transformed = lambda source=source: self.shift_scale_latents(args, source())
                    if hasattr(source, "adv_grid_key"):
                        transformed.adv_grid_key = source.adv_grid_key
                    transformed.adv_weight = getattr(source, "adv_weight", 1)
                    transformed_sources.append(transformed)
                sources = transformed_sources
                references = adv_cache.build_adv_references(
                    sources,
                    levels=self.adv_levels,
                    token_budget=self.adv_tokens_per_band,
                    reference_samples=self.adv_reference_samples,
                    features=self.adv_features,
                    bandwidths=self.adv_bandwidths,
                    seed=self.adv_rff_seed,
                    device=accelerator.device,
                )
                adv_cache.save_adv_cache(cache_path, signature, references)
                logger.info(f"Built stationary adv reference cache for {len(references)} latent grids: {cache_path}")
        accelerator.wait_for_everyone()
        self.adv_references = adv_cache.load_adv_cache(cache_path, signature, accelerator.device)

    def _apply_adv_term(self, final_loss, model_pred, clean, timesteps, weighting, noise_scheduler, args, batch):
        """Add the weighted fixed-reference MMD contribution to a scalar training loss."""
        self.adv_loss_value = None
        adv_needs_noisy = self.adv_prediction_mode != "x0_direct"
        noisy = self._adv_noisy_latents
        if adv_needs_noisy and noisy is None:
            if not self._adv_warned_missing:
                self._adv_warned_missing = True
                logger.warning(
                    "adv_scale > 0 but this trainer/path did not store noisy latents; "
                    "the stationary distribution-matching loss is INERT for this configuration."
                )
            return final_loss
        if self.adv_references is None:
            if not self._adv_warned_missing:
                self._adv_warned_missing = True
                logger.warning("adv_scale > 0 but no stationary reference cache is loaded; skipping the adv loss.")
            return final_loss

        x0_pred = hf_token_loss.hf_x0_hat(
            model_pred,
            noisy,
            timesteps,
            self.adv_prediction_mode,
            noise_scheduler=noise_scheduler,
            timesteps_in_sigma=self.adv_timesteps_in_sigma,
            eps_train=self.hf_eps_train,
        ).to(dtype=clean.dtype)
        adv_per_sample = dist_match.adv_per_sample_loss(
            x0_pred,
            self.adv_references,
            levels=self.adv_levels,
            tokens_per_band=self.adv_tokens_per_band,
        )
        if self.adv_snr_weighting:
            adv_per_sample = adv_per_sample * dist_match.adv_snr_weights(
                timesteps,
                self.adv_prediction_mode,
                noise_scheduler=noise_scheduler,
                timesteps_in_sigma=self.adv_timesteps_in_sigma,
            )
        if weighting is not None:
            adv_weight = weighting.detach().reshape(-1)
            if adv_weight.shape[0] == adv_per_sample.shape[0]:
                adv_per_sample = adv_per_sample * adv_weight.to(adv_per_sample.dtype)
        loss_weights = batch.get("loss_weights")
        if loss_weights is not None:
            adv_per_sample = adv_per_sample * loss_weights.reshape(-1).to(adv_per_sample.dtype)
        adv_per_sample = self.post_process_loss(adv_per_sample, args, timesteps, noise_scheduler)
        adv_term = adv_per_sample.mean()
        final_loss = final_loss + self.adv_scale * adv_term
        self.adv_loss_value = (self.adv_scale * adv_term).detach()
        return final_loss

    def build_adaptive_model_fn(self, unet, accelerator, weight_dtype):
        """Build a model_fn(noisy_latents, timesteps, wdtype) -> noise_pred for Algorithm 2.

        The returned closure captures the last training step's text_conds, args, batch,
        and masks.  When called with an *arbitrary* batch size N (e.g. chunk_len for the
        queue, or B*|S| for the batch-loss cache), it expands the captured conditioning
        tensors so their leading dimension matches N.

        Subclasses (e.g. AnimaNetworkTrainer) override this to call their own model
        instead of ``self.call_unet``.
        """
        text_conds = self._adaptive_last_text_conds
        text_masks = text_conds[1] if len(text_conds) > 1 else None
        adaptive_args = self._adaptive_last_args
        adaptive_batch = self._adaptive_last_batch if self._adaptive_last_batch is not None else {}
        base_batch_size = self._adaptive_last_latents.shape[0] if self._adaptive_last_latents is not None else 1

        def model_fn(noisy_latents, timesteps, wdtype):
            N = noisy_latents.shape[0]
            noisy_latents_in = noisy_latents.to(wdtype)

            # Expand conditioning to match the batch dimension of noisy_latents.
            # The queue path (single x_0) and batch-loss path (B*|S|) both produce
            # latents whose leading dimension may differ from the training batch B.
            if N != base_batch_size:
                # Repeat the first element to match N — safe because the adaptive
                # sampler always operates on a single x_0 expanded to N copies, or
                # on the full batch expanded by |S| copies per sample.
                expanded_conds = []
                for c in text_conds:
                    if isinstance(c, torch.Tensor) and c.shape[0] > 0:
                        expanded_conds.append(c[:1].expand(N, *c.shape[1:]).contiguous())
                    else:
                        expanded_conds.append(c)
                encoder_mask_bias = expanded_conds[1] if len(expanded_conds) > 1 else None
                # Autocast to match the training forward pass (avoids float32-vs-bf16
                # dtype mismatch in mixed-precision training).
                with accelerator.autocast():
                    return self.call_unet(
                        adaptive_args, accelerator, unet, noisy_latents_in, timesteps,
                        expanded_conds, encoder_mask_bias, adaptive_batch, wdtype,
                    )
            else:
                encoder_mask_bias = text_masks
                with accelerator.autocast():
                    return self.call_unet(
                        adaptive_args, accelerator, unet, noisy_latents_in, timesteps,
                        text_conds, encoder_mask_bias, adaptive_batch, wdtype,
                    )

        return model_fn

    def compute_adaptive_delta_before_step(self, unet, noise_scheduler, weight_dtype, accelerator, global_step):
        """Compute per-timestep losses with theta_k (before optimizer step) for Algorithm 2.

        For a single x_0 (used to build the queue), compute losses at ALL T timesteps.
        For the full batch, cache losses at the current |S| timesteps so the post-step
        hook can compute the full-batch delta at those same |S| timesteps.
        """
        if self.adaptive_manager is None:
            return
        if not self.adaptive_manager.should_update(global_step):
            return

        latents = self._adaptive_last_latents
        noise = self._adaptive_last_noise
        text_conds = self._adaptive_last_text_conds
        if latents is None or noise is None or text_conds is None:
            return

        model_fn = self.build_adaptive_model_fn(unet, accelerator, weight_dtype)

        # Compute per-timestep losses for a single x_0 at all T timesteps (for the queue)
        self._adaptive_losses_before = self.adaptive_manager.compute_per_timestep_losses(
            latents, noise, model_fn, weight_dtype, label="theta_k pre-step"
        )

        # Cache per-timestep losses for the FULL batch at the current |S| timesteps,
        # so that after the optimizer step we can compute the delta for the full batch
        # at those same |S| timesteps (Algorithm 2, line 7).
        self.adaptive_manager.cache_batch_losses_at_S(
            latents, noise, model_fn, weight_dtype
        )

    def compute_adaptive_delta_after_step(self, unet, noise_scheduler, weight_dtype, accelerator, network):
        """Run Algorithm 2 after optimizer step: compute delta and update sampler.

        Uses the full batch at the |S| selected timesteps (if a previous selection
        exists) to compute the delta, falling back to the single x_0 otherwise.
        """
        if self.adaptive_manager is None or self._adaptive_losses_before is None:
            return

        latents = self._adaptive_last_latents
        noise = self._adaptive_last_noise
        text_conds = self._adaptive_last_text_conds
        if latents is None or noise is None or text_conds is None:
            return

        model_fn = self.build_adaptive_model_fn(unet, accelerator, weight_dtype)

        # Algorithm 2: compute delta approximation. When a previous |S| selection
        # exists, the full batch at those timesteps is used (paper Algorithm 2 line 7).
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
        # to prevent permanently elevated VRAM (the sweeps spike peak reserved memory).
        if not self._adaptive_disable_empty_cache and accelerator.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_adaptive_model_type(self, args) -> str:
        """Return the model type for the adaptive timestep sampler.

        Subclasses should override this to return 'flow_matching' for
        flow-matching architectures (Flux, SD3, Lumina, Anima, etc.).
        """
        return "ddpm"

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)

    def update_metadata(self, metadata, args):
        pass

    def is_text_encoder_not_needed_for_training(self, args):
        return False  # use for sample images

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # set top parameter requires_grad = True for gradient checkpointing works
        text_encoder.text_model.embeddings.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        return accelerator.prepare(unet)

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train: bool = True):
        pass

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        pass

    def setup_tlora_masking(self, net_kwargs, network_dim, noise_scheduler):
        """
        Initialize T-LoRA timestep masking.

        Supports two paths:
        1. ``algo="tlora"`` — full TLoraModule with SVD-based parameterization
           (module-level mask from lycoris.modules.tlora).
        2. ``algo="locon"/"lora"`` + ``use_timestep_mask=True`` — lightweight
           mask flag on LoConModule, saves as standard lora_up/lora_down.

        Reads tlora_min_rank / tlora_mask_alpha (path 1) or
        tlora_min_rank / tlora_alpha (path 2) from network_args.
        Must be called after the network is created.
        """
        algo = (net_kwargs.get("algo", "lora") or "lora").lower()

        # Path 2: LoCon flag approach (algo=lora/locon + use_timestep_mask=True)
        if algo in ("lora", "locon", "ortholora") and net_kwargs.get("use_timestep_mask"):
            self.tlora_enabled = True
            self.tlora_use_network_method = True
            self.tlora_max_rank = int(network_dim) if network_dim is not None else 4
            tlora_min_rank = net_kwargs.get("tlora_min_rank", None)
            if tlora_min_rank is None:
                self.tlora_min_rank = max(1, self.tlora_max_rank // 2)
            else:
                self.tlora_min_rank = int(tlora_min_rank)
            self.tlora_min_rank = max(0, min(self.tlora_min_rank, self.tlora_max_rank))
            self.tlora_mask_alpha = float(net_kwargs.get("tlora_alpha", 1.0))
            self.tlora_max_timestep = noise_scheduler.config.num_train_timesteps
            logger.info(
                f"T-LoRA masking enabled (LoCon flag): max_rank={self.tlora_max_rank}, "
                f"min_rank={self.tlora_min_rank}, alpha={self.tlora_mask_alpha}, "
                f"max_timestep={self.tlora_max_timestep}"
            )
            return

        # Path 1: Full TLoraModule approach (algo=tlora)
        if algo != "tlora":
            return
        if not TLORA_AVAILABLE:
            logger.warning("T-LoRA requested but lyco_tlora is not available. Skipping T-LoRA setup.")
            return

        self.tlora_enabled = True
        self.tlora_use_network_method = False
        self.tlora_max_rank = int(network_dim) if network_dim is not None else 4
        tlora_min_rank = net_kwargs.get("tlora_min_rank", None)
        if tlora_min_rank is None:
            self.tlora_min_rank = int(math.ceil(self.tlora_max_rank * 0.5))
        else:
            self.tlora_min_rank = int(tlora_min_rank)
        self.tlora_min_rank = max(0, min(self.tlora_min_rank, self.tlora_max_rank))
        self.tlora_mask_alpha = float(net_kwargs.get("tlora_mask_alpha", 1.0))
        self.tlora_max_timestep = noise_scheduler.config.num_train_timesteps
        logger.info(
            f"T-LoRA masking enabled (TLoraModule): max_rank={self.tlora_max_rank}, "
            f"min_rank={self.tlora_min_rank}, mask_alpha={self.tlora_mask_alpha}, "
            f"max_timestep={self.tlora_max_timestep}"
        )

    def apply_tlora_mask(self, timesteps: torch.Tensor):
        """
        Compute and set the T-LoRA timestep mask for the current batch.

        Dispatches to either the network-level method (LoCon flag approach,
        which uses a shared GPU buffer aliased across all modules) or the
        module-level functions from lycoris.modules.tlora (full TLoraModule).
        """
        if not self.tlora_enabled:
            return

        if self.tlora_use_network_method:
            # LoCon flag: use network.set_timestep_mask (shared buffer, no CPU transfer)
            self.network.set_timestep_mask(timesteps, self.tlora_max_timestep)
        else:
            # Full TLoraModule: compute mask tensor, set via module-level function
            if timesteps.numel() == 1:
                mask = compute_timestep_mask(
                    timestep=int(timesteps.item()),
                    max_timestep=self.tlora_max_timestep,
                    max_rank=self.tlora_max_rank,
                    min_rank=self.tlora_min_rank,
                    alpha=self.tlora_mask_alpha,
                )
            else:
                mask = compute_timestep_mask_batch(
                    timesteps=timesteps,
                    max_timestep=self.tlora_max_timestep,
                    max_rank=self.tlora_max_rank,
                    min_rank=self.tlora_min_rank,
                    alpha=self.tlora_mask_alpha,
                )
            set_timestep_mask(mask)

    def clear_tlora_mask_if_needed(self):
        """Clear the T-LoRA mask after the forward pass."""
        if not self.tlora_enabled:
            return
        if self.tlora_use_network_method:
            self.network.clear_timestep_mask()
        else:
            clear_timestep_mask()

    def setup_lora2_regularization(self, net_kwargs):
        """
        Initialize LoRA² rank regularization if the algo is lora2.

        Reads lora2_lambda_r from network_args.
        Must be called after the network is created.
        """
        algo = (net_kwargs.get("algo", "lora") or "lora").lower()
        if algo != "lora2":
            return
        if not LORA2_AVAILABLE:
            logger.warning("LoRA² requested but LoRA2Module is not available. Skipping LoRA² setup.")
            return

        self.lora2_enabled = True
        self.lora2_lambda_r = float(net_kwargs.get("lora2_lambda_r", 1e-4))
        logger.info(
            f"LoRA² rank regularization enabled: lambda_r={self.lora2_lambda_r}"
        )

    # endregion

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
        edm2_model=None
    ) -> torch.Tensor:
        """
        Process a batch for the network
        """
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(device=accelerator.device))
            else:
                # latentに変換
                if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                    latents = self.encode_images_to_latents(args, vae, batch["images"].to(device=accelerator.device, dtype=vae_dtype))
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size] for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = self.encode_images_to_latents(args, vae, chunk.to(device=accelerator.device, dtype=vae_dtype))
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                # NaNが含まれていれば警告を表示し0に置き換える
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(args, latents)

            # Prepare inpainting masked_latents if batch contains masks
            if batch.get("masks") is not None and batch.get("masked_images") is not None:
                masked_latents = self.encode_images_to_latents(
                    args, vae, batch["masked_images"].to(accelerator.device, dtype=vae_dtype)
                )
                batch["masked_latents"] = self.shift_scale_latents(args, masked_latents)

        text_encoder_conds = []
        masks_reshaped = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs
            if isinstance(text_encoding_strategy, SdxlTextEncodingStrategy):
                masks_reshaped = text_encoder_outputs_list[3:]

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            # TODO this does not work if 'some text_encoders are trained' and 'some are not and not cached'
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                # Get the text embedding for conditioning
                if args.weighted_captions:
                    input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(args, accelerator, text_encoders),
                        input_ids_list,
                        weights_list,
                    )
                else:
                    input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                    if isinstance(text_encoding_strategy, SdxlTextEncodingStrategy):
                        masks = [mask.to(accelerator.device) for mask in batch["attn_mask_list"]]
                        encoded_text_encoder_conds, masks_reshaped = text_encoding_strategy.encode_tokens(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(args, accelerator, text_encoders),
                            input_ids,
                            attn_masks=masks,
                        )
                    else:
                        encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(args, accelerator, text_encoders),
                            input_ids
                        )
                if args.full_fp16:
                    encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

            # if text_encoder_conds is not cached, use encoded_text_encoder_conds
            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                # if encoded_text_encoder_conds is not None, update cached text_encoder_conds
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        # Store text encoder conditions and batch for adaptive Algorithm 2
        if self.adaptive_manager is not None and is_train and self._adaptive_update_pending:
            self._adaptive_last_text_conds = [c.detach() if isinstance(c, torch.Tensor) else c for c in text_encoder_conds]
            self._adaptive_last_batch = batch

        # sample noise, call unet, get target
        noise_pred, target, timesteps, weighting, noise = self.get_noise_pred_and_target(
            args,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            masks_reshaped,
            unet,
            network,
            weight_dtype,
            train_unet,
            is_train=is_train,
        )

        # Cast to float64 (Double Precision) for Grokking
        noise_pred = noise_pred.to(dtype=torch.float64)
        target = target.to(dtype=torch.float64)

        if is_train:
            if args.differential_guidance:
                target = noise_pred + (float(args.differential_guidance_scale) * (target - noise_pred))

            huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
            loss = train_util.conditional_loss(noise_pred, target, args.loss_type, "none", huber_c, scale=float(args.loss_scale))
            if weighting is not None:
                loss = loss * weighting

            if args.contrastive_flow_matching and latents.size(0) > 1:
                # CRITICAL FIX: Add .detach() to prevent gradients flowing through negative samples
                negative_latents = latents.roll(1, 0).detach()
                negative_noise = noise.roll(1, 0).detach()
                with torch.no_grad():
                    if getattr(args, "flow_model", False) or getattr(args, "_anima_model", False):
                        target_negative = negative_noise - negative_latents
                    else:
                        target_negative = noise_scheduler.get_velocity(negative_latents, negative_noise, timesteps)

                # Handle cast for CFM
                target_negative = target_negative.to(dtype=torch.float64)

                loss_contrastive = torch.nn.functional.mse_loss(
                    noise_pred, target_negative, reduction="none"
                )
                # Store CFM component for logging (before applying lambda)
                #loss_cfm = loss_contrastive.mean([1, 2, 3]).mean().detach()
                loss = loss - float(args.cfm_lambda) * loss_contrastive
            if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                loss = apply_masked_loss(loss, batch)

            # --- LWD Wavelet Masking: apply spatial mask to element-wise loss ---
            if self.wavelet_masking_enabled and self._noisy_latents is not None:
                with torch.no_grad():
                    A = train_util.compute_wavelet_attention_map(self._noisy_latents, self.wavelet_dwt)
                    # Flow-matching trainers (Flux/SD3/Anima/Lumina/Hunyuan) return
                    # timesteps in [0, 1]; DDPM trainers return them in [0, T].
                    # get_adaptive_model_type() is the explicit discriminator
                    # (overridden to "flow_matching" by all flow trainers, defaults
                    # to "ddpm" in the base class).
                    is_flow_matching = self.get_adaptive_model_type(args) == "flow_matching"
                    M = train_util.get_wavelet_mask(
                        A,
                        l=float(getattr(args, "wavelet_mask_l_bound", 0.3)),
                        T=noise_scheduler.config.num_train_timesteps,
                        timesteps=timesteps,
                        flow_matching=is_flow_matching,
                    )
                # M shape: (B, 1, H, W), loss shape: (B, C, H, W) or (B, seq_len)
                if loss.ndim == 4 and loss.shape[2:] == M.shape[2:]:
                    loss = loss * M
                    self._wavelet_mask_ratio = M.mean().item()
                elif loss.ndim == 2:
                    # For packed/sequence models, skip masking (shape mismatch)
                    self._wavelet_mask_ratio = 0.0
                else:
                    logger.warning_once(
                        f"Wavelet mask shape {M.shape} incompatible with loss shape {loss.shape}, skipping mask"
                    )
                    self._wavelet_mask_ratio = 0.0
        else:
                loss = train_util.conditional_loss(noise_pred, target, "l2", "none", None)

        # --- Token-level hard mining: reweight spatial tokens by detached per-token difficulty ---
        if is_train and getattr(args, "token_mining", False):
            mining_sigmas = None
            if not getattr(args, "token_mining_no_sigma_gate", False):
                ts = timesteps.detach().float().reshape(-1)
                if ts.numel() > 0 and ts.max() > 1.5:  # discrete timesteps in [0, T]
                    mining_sigmas = ts / noise_scheduler.config.num_train_timesteps
                else:  # flow-matching trainers already return sigmas in [0, 1]
                    mining_sigmas = ts
            loss = custom_train_functions.apply_token_mining(
                loss,
                sigmas=mining_sigmas,
                alpha=float(getattr(args, "token_mining_alpha", 1.0)),
                min_weight=float(getattr(args, "token_mining_min_weight", 0.25)),
                max_weight=float(getattr(args, "token_mining_max_weight", 4.0)),
                sigma_gate=mining_sigmas is not None,
            )

        loss = loss.mean(dim=list(range(1, loss.ndim)))  # mean over all dims except batch

        if is_train:
            loss_weights = batch["loss_weights"]  # 各sampleごとのweight
            loss = loss * loss_weights
            loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

        if is_train and args.loss_multiplier:
            loss.mul_(float(args.loss_multiplier) if args.loss_multiplier is not None else 1.0)

        # For logging
        pre_scaling_loss = loss.mean()

        if is_train and args.edm2_loss_weighting:
            loss, loss_scaled = edm2_model(loss, timesteps)
            loss_scaled = loss_scaled.mean()
        else:
            loss_scaled = None

        final_loss = loss.mean()

        # LoRA²: add rank regularization loss
        if is_train and self.lora2_enabled and LORA2_AVAILABLE:
            rank_reg_loss = LoRA2Module.get_total_rank_reg_loss()
            if rank_reg_loss.item() > 0:
                final_loss = final_loss + self.lora2_lambda_r * rank_reg_loss

        # Focal Frequency Loss: auxiliary frequency-domain loss on latent space
        self.ffl_loss_value = None
        if is_train and self.ffl_enabled and self.ffl_module is not None:
            ffl_loss = self.ffl_module(noise_pred, target)
            self.ffl_loss_value = ffl_loss.detach().item()
            final_loss = final_loss + ffl_loss

        # Patch Topology Loss: VAE-free spatial self-similarity topology matching
        # Supports delayed start (--patch_topology_start_step), linear warmup
        # (--patch_topology_warmup_steps), optional dynamic multi-loss weighting
        # (--patch_topology_dynamic_weighting: none/dwa/gradnorm), spatial mask
        # weighting (masked loss / alpha masks) and per-sample loss_weights for
        # consistency with the base objective.
        self.patch_topology_loss_value = None
        self.patch_topology_effective_weight = None
        if is_train and self.patch_topology_enabled and self.patch_topology_loss_module is not None:
            # Warmup gate: ramps linearly from 0 to 1 over warmup_steps after start_step.
            warmup_gate = 0.0
            if self._patch_topology_current_step >= self.patch_topology_start_step:
                if self.patch_topology_warmup_steps > 0:
                    steps_into_warmup = self._patch_topology_current_step - self.patch_topology_start_step
                    warmup_gate = min(1.0, float(steps_into_warmup) / float(self.patch_topology_warmup_steps))
                else:
                    warmup_gate = 1.0

            if warmup_gate > 0.0:
                # Spatial mask for masked/inpainting training, mirroring apply_masked_loss.
                topo_mask = None
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    if noise_pred.ndim == 4:
                        topo_mask = extract_spatial_mask(
                            batch, noise_pred.shape[2:], noise_pred.device, torch.float32
                        )

                try:
                    patch_topo_loss_per_sample = self.patch_topology_loss_module(
                        pred=noise_pred,
                        target=target,
                        timesteps=timesteps,
                        mask=topo_mask,
                    )
                except ValueError as e:
                    # e.g. non-square sequence lengths from packed DiT outputs
                    logger.warning_once(f"Patch Topology Loss skipped for this batch: {e}")
                    patch_topo_loss_per_sample = None

                if patch_topo_loss_per_sample is not None:
                    # Per-sample loss_weights, consistent with the base loss.
                    loss_weights = batch.get("loss_weights")
                    if loss_weights is not None:
                        patch_topo_loss_per_sample = patch_topo_loss_per_sample * loss_weights.to(
                            patch_topo_loss_per_sample.dtype
                        )

                    patch_topo_loss_mean = patch_topo_loss_per_sample.mean()

                    # Effective weight: dynamic multi-loss weighting (dwa/gradnorm) or static.
                    if self.patch_topology_weighter is not None:
                        shared_params = None
                        if self.patch_topology_weighter.mode == "gradnorm" and network is not None:
                            # GradNorm balances gradient norms on shared (trainable) parameters;
                            # restrict to the last few LoRA tensors to bound the extra backward cost.
                            trainable = [p for p in network.parameters() if p.requires_grad]
                            shared_params = trainable[-8:] if trainable else None
                        dynamic_weight = self.patch_topology_weighter.compute_weight(
                            final_loss, patch_topo_loss_mean, shared_params=shared_params
                        )
                        effective_weight = warmup_gate * dynamic_weight
                    else:
                        effective_weight = warmup_gate * self.patch_topology_full_weight

                    self.patch_topology_loss_value = patch_topo_loss_mean.detach().item()
                    self.patch_topology_effective_weight = effective_weight
                    final_loss = final_loss + effective_weight * patch_topo_loss_mean

        # --- High-Frequency Token latent loss: per-token x0-MSE weighted by clean-token detail ---
        # Opt-in auxiliary (hf_scale > 0). The Python-level gate inside hf_apply_term makes
        # the off-mode bit-identical (no extra ops/allocations/RNG). Weights derive from the
        # clean target only (never from the prediction), and the term is differentiable only
        # through x0_hat. The detached scaled value is stored for logging (materialized at the
        # existing periodic sync, not on the hot path).
        self.hf_loss_value = None
        if is_train and self._hf_noisy_latents is not None:
            # HF-only one-sided high-noise gate. Low-noise samples remain exactly
            # at weight 1; the main loss weighting is not modified.
            hf_weighting = weighting
            if self.hf_prediction_mode in (
                "flow",
                "x0_residual_eps",
                "x0_direct",
                "vpred_ddpm",
                "eps_ddpm",
            ):
                hf_snr = hf_token_loss.hf_snr_from_timesteps(
                    timesteps,
                    mode=self.hf_prediction_mode,
                    noise_scheduler=noise_scheduler,
                    timesteps_in_sigma=self.hf_timesteps_in_sigma,
                )
                # Epsilon -> x0 reconstruction has a 1/SNR gradient factor;
                # a zero floor is required to cap that factor at low SNR.
                hf_min_gate = (
                    0.0 if self.hf_prediction_mode == "eps_ddpm" else self.hf_high_noise_min_weight
                )
                hf_gate = hf_token_loss.hf_one_sided_snr_gate(
                    hf_snr,
                    snr_cut=self.hf_high_noise_snr_cut,
                    min_gate=hf_min_gate,
                    power=self.hf_high_noise_power,
                )
                if hf_weighting is None:
                    hf_weighting = hf_gate
                else:
                    hf_weighting = (
                        hf_weighting.detach().reshape(-1).to(dtype=hf_gate.dtype) * hf_gate
                    )

            final_loss, self.hf_loss_value = hf_token_loss.hf_apply_term(
                final_loss,
                noise_pred,
                clean=latents,
                noisy=self._hf_noisy_latents,
                timesteps=timesteps,
                weighting=hf_weighting,
                scale=self.hf_scale,
                exponent=self.hf_exponent,
                patch=self.hf_patch,
                mode=self.hf_prediction_mode,
                noise_scheduler=noise_scheduler,
                timesteps_in_sigma=self.hf_timesteps_in_sigma,
                eps_train=self.hf_eps_train,
            )

        # --- Multiscale MSE x0-prediction ("anchor") loss: frequency-even Laplacian-pyramid MSE on the predicted x0 ---
        # Opt-in auxiliary (anchor_scale > 0). The Python-level gate keeps the off path
        # bit-identical (no extra ops/allocations/RNG). The pyramid is built ONCE on the
        # delta (x0_pred - clean); band MSEs are whitened by calibrated band filter
        # energies (cached per shape/dtype) and averaged uniformly. Batch aggregation
        # mirrors the primary loss's per-sample weighting and reduction (spec §3.8).
        self.anchor_loss_value = None
        if is_train and self.anchor_scale > 0.0 and self._anchor_noisy_latents is None and not self._anchor_warned_missing:
            # Trainers/paths that never store the noisy latents (e.g. flux/sd3/lumina/
            # hunyuan, which override get_noise_pred_and_target, and the iLECO/ADDifT
            # distillation paths, which have no analytic clean x0) silently skip the
            # term — make that visible instead of a silent no-op.
            self._anchor_warned_missing = True
            logger.warning(
                "anchor_scale > 0 but no noisy latents were stored by this trainer/path; "
                "the multiscale x0-prediction anchor loss is INERT for this configuration."
            )
        if is_train and self.anchor_scale > 0.0 and self._anchor_noisy_latents is not None:
            anchor_per_sample = anchor_loss.anchor_per_sample_from_prediction(
                noise_pred,
                clean=latents,
                noisy=self._anchor_noisy_latents,
                timesteps=timesteps,
                mode=self.anchor_prediction_mode,
                noise_scheduler=noise_scheduler,
                timesteps_in_sigma=self.anchor_timesteps_in_sigma,
                levels=self.anchor_levels,
            )  # [B] fp32, differentiable only through x0_pred

            # Optional soft SNR(t) gate (batch-mean-1 normalized; reshapes across t only).
            if self.anchor_snr_weighting:
                anchor_snr_w = anchor_loss.anchor_snr_weights(
                    timesteps,
                    self.anchor_prediction_mode,
                    noise_scheduler=noise_scheduler,
                    timesteps_in_sigma=self.anchor_timesteps_in_sigma,
                )
                anchor_per_sample = anchor_per_sample * anchor_snr_w

            # Mirror the primary loss's per-sample weighting: scheduler weighting,
            # per-sample data weights, then the per-sample timestep weighting
            # (min-SNR-gamma etc.) applied by post_process_loss.
            if weighting is not None:
                anchor_w = weighting.detach().reshape(-1)
                if anchor_w.shape[0] == anchor_per_sample.shape[0]:
                    anchor_per_sample = anchor_per_sample * anchor_w.to(anchor_per_sample.dtype)
            anchor_loss_weights = batch.get("loss_weights")
            if anchor_loss_weights is not None:
                anchor_per_sample = anchor_per_sample * anchor_loss_weights.reshape(-1).to(anchor_per_sample.dtype)
            anchor_per_sample = self.post_process_loss(anchor_per_sample, args, timesteps, noise_scheduler)

            anchor_term = anchor_per_sample.mean()
            final_loss = final_loss + self.anchor_scale * anchor_term
            # Detached, already-scaled contribution for logging (materialized at the
            # caller's existing periodic sync — no .item() on the hot path).
            self.anchor_loss_value = (self.anchor_scale * anchor_term).detach()

        # --- Stationary multi-scale distribution matching (fixed-reference RFF-MMD) ---
        if is_train and self.adv_scale > 0.0:
            final_loss = self._apply_adv_term(
                final_loss,
                noise_pred,
                latents,
                timesteps,
                weighting,
                noise_scheduler,
                args,
                batch,
            )

        return final_loss, pre_scaling_loss, loss_scaled
    
    def process_val_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        train_text_encoder=True,
        train_unet=True,
        timesteps_list: list = [50, 350, 500, 650, 950]
    ) -> torch.Tensor:
        """
        Process a batch for the network to determine val loss
        """
        total_loss = 0.0 
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(device=accelerator.device))
            else:
                # latentに変換
                if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                    latents = self.encode_images_to_latents(args, vae, batch["images"].to(device=accelerator.device, dtype=vae_dtype))
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size] for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        chunk = self.encode_images_to_latents(args, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                        list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                # NaNが含まれていれば警告を表示し0に置き換える
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(args, latents)

            text_encoder_conds = []
            masks_reshaped = []
            text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
            if text_encoder_outputs_list is not None:
                text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs
                if isinstance(text_encoding_strategy, SdxlTextEncodingStrategy):
                    masks_reshaped = text_encoder_outputs_list[3:]

            if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
                # TODO this does not work if 'some text_encoders are trained' and 'some are not and not cached'
                with accelerator.autocast():
                    # Get the text embedding for conditioning
                    if args.weighted_captions:
                        input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                        encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(args, accelerator, text_encoders),
                            input_ids_list,
                            weights_list,
                        )
                    else:
                        input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                        if isinstance(text_encoding_strategy, SdxlTextEncodingStrategy):
                            masks = [mask.to(accelerator.device) for mask in batch["attn_mask_list"]]
                            encoded_text_encoder_conds, masks_reshaped = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                self.get_models_for_text_encoding(args, accelerator, text_encoders),
                                input_ids,
                                attn_masks=masks,
                            )
                        else:
                            encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                self.get_models_for_text_encoding(args, accelerator, text_encoders),
                                input_ids
                            )
                    if args.full_fp16:
                        encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

                # if text_encoder_conds is not cached, use encoded_text_encoder_conds
                if len(text_encoder_conds) == 0:
                    text_encoder_conds = encoded_text_encoder_conds
                else:
                    # if encoded_text_encoder_conds is not None, update cached text_encoder_conds
                    for i in range(len(encoded_text_encoder_conds)):
                        if encoded_text_encoder_conds[i] is not None:
                            text_encoder_conds[i] = encoded_text_encoder_conds[i]

            batch_size = latents.shape[0]
            per_t_step_losses = []
            for fixed_timesteps in timesteps_list:
                timesteps = torch.full((batch_size,), fixed_timesteps, dtype=torch.long, device=latents.device)
                # sample noise, call unet, get target
                noise_pred, target, _, _, _ = self.get_noise_pred_and_target(
                    args,
                    accelerator,
                    noise_scheduler,
                    latents,
                    batch,
                    text_encoder_conds,
                    masks_reshaped,
                    unet,
                    network,
                    weight_dtype,
                    train_unet,
                    fixed_timesteps=timesteps,
                    is_train=False,
                )

                # Cast to float64 (Double Precision) for Grokking
                noise_pred = noise_pred.to(dtype=torch.float64)
                target = target.to(dtype=torch.float64)

                loss = train_util.conditional_loss(noise_pred, target, "l2", "none", None)
                loss = loss.mean(dim=list(range(1, loss.ndim)))  # mean over all dims except batch
                loss = loss.mean()
                total_loss += loss
                per_t_step_losses.append(loss)

        average_loss = total_loss / len(timesteps_list)
        per_timestep_losses = {t: l.detach().item() for t, l in zip(timesteps_list, per_t_step_losses)}

        return average_loss, per_timestep_losses

    def cast_text_encoder(self, args):
        return True  # default for other than HunyuanImage

    def cast_vae(self, args):
        return True  # default for other than HunyuanImage

    def cast_unet(self, args):
        return True  # default for other than HunyuanImage

    def switch_rng_state(self, val_seed: int, accelerator) -> tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]:
        cpu_rng_state = torch.get_rng_state()
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        if accelerator.device.type == "cuda":
            gpu_rng_state = torch.cuda.get_rng_state()
        elif accelerator.device.type == "xpu":
            gpu_rng_state = torch.xpu.get_rng_state()
        elif accelerator.device.type == "mps":
            gpu_rng_state = torch.cuda.get_rng_state()
        else:
            gpu_rng_state = None

        random.seed(val_seed)
        np.random.seed(val_seed)
        torch.manual_seed(val_seed)
        if accelerator.device.type == "cuda":
            torch.cuda.manual_seed_all(val_seed)

        return (cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state)

    def restore_rng_state(self, rng_states: tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple], accelerator):
        cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state = rng_states
        torch.set_rng_state(cpu_rng_state)
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        if gpu_rng_state is not None:
            if accelerator.device.type == "cuda":
                torch.cuda.set_rng_state(gpu_rng_state)
            elif accelerator.device.type == "xpu":
                torch.xpu.set_rng_state(gpu_rng_state)
            elif accelerator.device.type == "mps":
                torch.cuda.set_rng_state(gpu_rng_state)

    def calculate_val_loss(self, 
                           global_step,
                           epoch_step,
                           train_dataloader,
                           val_loss_recorder,
                           val_dataloader,
                           cyclic_val_dataloader,
                           network, 
                           tokenize_strategy, 
                           text_encoders, 
                           text_encoding_strategy, 
                           unet, 
                           vae, 
                           noise_scheduler, 
                           vae_dtype, 
                           weight_dtype, 
                           accelerator, 
                           args, 
                           epoch,
                           batch=None,
                           train_text_encoder=True):
        if not train_util.calculate_val_loss_check(args,global_step,epoch_step,val_dataloader,train_dataloader):
            return None, None, None
        
        if batch is not None:
            self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=False)
   
        rng_states = self.switch_rng_state(int(args.validation_seed) if args.validation_seed else 23, accelerator)

        timesteps_list = ast.literal_eval(args.validation_timesteps)
              
        accelerator.print("") 
        accelerator.print("Validating バリデーション処理...")
        total_loss = 0.0
        total_samples = 0
        per_timestep_total_loss = {}
        per_timestep_total_samples = {}
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

                # Determine current batch size for proper weighted averaging
                if "latents" in batch and batch["latents"] is not None:
                    current_batch_size = batch["latents"].shape[0]
                elif "images" in batch:
                    current_batch_size = batch["images"].shape[0]
                elif "captions" in batch:
                    current_batch_size = len(batch["captions"])
                else:
                    current_batch_size = 1

                loss, batch_per_t_losses = self.process_val_batch(batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype,
                                              weight_dtype, accelerator, args, text_encoding_strategy, tokenize_strategy,
                                              train_text_encoder=train_text_encoder,
                                              timesteps_list=timesteps_list)
                total_loss += loss.detach().item() * current_batch_size
                total_samples += current_batch_size
                # Accumulate per-timestep losses
                for t, t_loss in batch_per_t_losses.items():
                    per_timestep_total_loss[t] = per_timestep_total_loss.get(t, 0.0) + t_loss * current_batch_size
                    per_timestep_total_samples[t] = per_timestep_total_samples.get(t, 0) + current_batch_size
            current_val_loss = total_loss / total_samples if total_samples > 0 else 0.0
            val_loss_recorder.add(current_val_loss)

        average_val_loss: float = val_loss_recorder.average
        # Compute per-timestep average validation losses
        per_timestep_avg = {
            f"loss/val/t{int(t)}": per_timestep_total_loss[t] / per_timestep_total_samples[t]
            for t in per_timestep_total_loss
        }
        logs = {"loss/current_val_loss": current_val_loss, "loss/average_val_loss": average_val_loss, **per_timestep_avg}

        self.restore_rng_state(rng_states, accelerator)

        # Release CUDA caching-allocator reserved memory from validation forward
        # passes (multiple timesteps × batches accumulate distinct tensor-size
        # pools that the allocator never frees on its own).
        clean_memory_on_device(accelerator.device)

        return current_val_loss, average_val_loss, logs


    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        self.setup_hf_objective(args)  # High-Frequency Token latent loss (validates hf_scale/hf_exponent/hf_patch)
        self.setup_anchor_objective(args)  # Multiscale MSE x0-prediction anchor loss (validates anchor args)
        self.setup_adv_objective(args)  # Stationary multi-scale RFF-MMD distribution matching
        train_util.set_torch_cuda_reduced_precision(args)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)

        if getattr(args, "flow_model", False):
            logger.info("Using Rectified Flow training objective.")
            if args.v_parameterization:
                raise ValueError("`--flow_model` is incompatible with `--v_parameterization`; Rectified Flow already predicts velocity.")
            if args.min_snr_gamma:
                logger.info("`--min_snr_gamma` is enabled for Rectified Flow (flow-matching SNR adaptation).")
            if args.debiased_estimation_loss:
                logger.warning("`--debiased_estimation_loss` is ignored when Rectified Flow is enabled.")
                args.debiased_estimation_loss = False
            if args.scale_v_pred_loss_like_noise_pred:
                logger.warning("`--scale_v_pred_loss_like_noise_pred` is ignored when Rectified Flow is enabled.")
                args.scale_v_pred_loss_like_noise_pred = False
            if args.v_pred_like_loss:
                logger.warning("`--v_pred_like_loss` is ignored when Rectified Flow is enabled.")
                args.v_pred_like_loss = None
            if args.flow_use_ot:
                logger.info("Using cosine optimal transport pairing for Rectified Flow batches.")
                
            shift_enabled = args.flow_uniform_shift or args.flow_uniform_static_ratio is not None
            distribution = getattr(args, "flow_timestep_distribution", "logit_normal")
            if distribution == "logit_normal":
                flow_logit_std = float(getattr(args, "flow_logit_std", 1.0))
                flow_logit_mean = float(getattr(args, "flow_logit_mean", 0.0))
                if flow_logit_std == 0:
                    raise ValueError("`--flow_logit_std` must be non-zero.")
                logger.info(
                    "Rectified Flow timesteps sampled from logit-normal distribution with "
                    f"mean={flow_logit_mean}, std={flow_logit_std}."
                )
            elif distribution == "uniform":
                logger.info("Rectified Flow timesteps sampled uniformly in [0, 1].")
            else:
                raise ValueError(f"Unknown Rectified Flow timestep distribution: {distribution}")

            if shift_enabled:
                if args.flow_uniform_static_ratio is not None:
                    flow_uniform_static_ratio = float(getattr(args, "flow_uniform_static_ratio", 0.0))
                    if flow_uniform_static_ratio <= 0:
                        raise ValueError("`--flow_uniform_static_ratio` must be positive.")
                    logger.info(
                        f"Rectified Flow timestep shift uses static ratio={flow_uniform_static_ratio}."
                    )
                else:
                    logger.info(
                        f"Rectified Flow timestep shift uses base pixels={args.flow_uniform_base_pixels}."
                    )

        if args.contrastive_flow_matching and not (args.v_parameterization or getattr(args, "flow_model", False) or getattr(args, "_anima_model", False)):
            raise ValueError("`--contrastive_flow_matching` requires either v-parameterization or Rectified Flow.")

        if getattr(args, "vae_custom_scale", None) is not None:
            try:
                self.vae_scale_factor = float(args.vae_custom_scale)
            except (TypeError, ValueError):
                raise ValueError("`--vae_custom_scale` must be a valid number")
            logger.info(f"Using custom VAE scale factor: {self.vae_scale_factor}")
        if getattr(args, "vae_custom_shift", None) is not None:
            try:
                self.latent_shift = float(args.vae_custom_shift)
            except (TypeError, ValueError):
                raise ValueError("`--vae_custom_shift` must be a valid number")
            logger.info(f"Using custom VAE shift factor: {self.latent_shift}")
        else:
            self.latent_shift = 0.0

        args.vae_scale_factor = self.vae_scale_factor
        args.vae_shift_factor = self.latent_shift

        if self.adv_scale > 0.0:
            if int(getattr(args, "cache_aug_refresh_epochs", 0) or 0) > 0:
                raise ValueError("adv_scale > 0 is incompatible with epoch-refreshed latent augmentation caches")
            if not args.cache_latents:
                logger.info("adv_scale > 0 requires stable clean reference samples; enabling latent caching")
                args.cache_latents = True
        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        train_util.args_set_seed(args)

        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)  # will be removed after sample_image is refactored

        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # データセットを準備する
        if args.dataset_class is None:
            support_controlnet_dataset = args.masked_loss or getattr(args, "addift", False)
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, support_controlnet_dataset, True))
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
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
            # use arbitrary dataset class
            train_dataset_group = train_util.load_arbitrary_dataset(args)
            val_dataset_group = None  # placeholder until validation dataset supported for arbitrary

        if args.protected_tags_file:
            logger.info("Injecting protected_tags_file into datasets...")
            for ds in train_dataset_group.datasets:
                ds.protected_tags_file = args.protected_tags_file
        if args.log_caption_tag_dropout:
            logger.info("Enabling caption tag dropout logging for datasets...")
            for ds in train_dataset_group.datasets:
                ds.log_caption_tag_dropout = True
        if args.log_caption_dropout:
            logger.info("Enabling caption dropout logging for datasets...")
            for ds in train_dataset_group.datasets:
                ds.log_caption_dropout = True

        # K-variant sampled augmentation caching: precompute augmented latent/caption variants.
        # Must be set before the cacheability assertions below (they depend on the variant config).
        latents_aug_variants = int(getattr(args, "cache_aug_variants", 0) or 0)
        caption_aug_variants = int(getattr(args, "cache_caption_variants", 0) or 0)
        if latents_aug_variants > 1 or caption_aug_variants > 1:
            if hasattr(train_dataset_group, "set_aug_variant_config"):
                train_dataset_group.set_aug_variant_config(latents_aug_variants, caption_aug_variants)
            if val_dataset_group is not None and hasattr(val_dataset_group, "set_aug_variant_config"):
                val_dataset_group.set_aug_variant_config(latents_aug_variants, caption_aug_variants)

            if latents_aug_variants > 1:
                if not cache_latents:
                    logger.warning("--cache_aug_variants has no effect without --cache_latents / --cache_latentsがないため--cache_aug_variantsは無効です")
                else:
                    logger.info(f"caching up to {latents_aug_variants} latent augmentation variants per image.")
            if caption_aug_variants > 1:
                if not getattr(args, "cache_text_encoder_outputs", False):
                    logger.warning(
                        "--cache_caption_variants has no effect without --cache_text_encoder_outputs / --cache_text_encoder_outputsがないため--cache_caption_variantsは無効です"
                    )
                elif getattr(args, "weighted_captions", False):
                    logger.warning(
                        "--cache_caption_variants is ignored with --weighted_captions (captions are tokenized per step) / --weighted_captions使用時は--cache_caption_variantsは無視されます"
                    )
                else:
                    logger.info(f"caching up to {caption_aug_variants} caption variants per image.")

        # Epoch-variant refresh configuration
        aug_refresh_epochs = int(getattr(args, "cache_aug_refresh_epochs", 0) or 0)
        if aug_refresh_epochs > 0:
            if latents_aug_variants <= 1 and caption_aug_variants <= 1:
                logger.warning("--cache_aug_refresh_epochs requires --cache_aug_variants or --cache_caption_variants > 1; ignoring")
                aug_refresh_epochs = 0
            else:
                if hasattr(train_dataset_group, "set_aug_refresh_epochs"):
                    train_dataset_group.set_aug_refresh_epochs(aug_refresh_epochs)
                logger.info(f"augmentation variants will be refreshed every {aug_refresh_epochs} epoch(s) in-memory.")

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            train_util.debug_dataset(train_dataset_group)

            if val_dataset_group is not None:
                val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
                train_util.debug_dataset(val_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used (use --cache_aug_variants to cache augmented variants) / latentをキャッシュするときはcolor_augとrandom_cropは使えません（--cache_aug_variantsでaugmentation済みバリアントをキャッシュ可能です）"
            if val_dataset_group is not None:
                assert (
                    val_dataset_group.is_latent_cacheable()
                ), "when caching latents, either color_aug or random_crop cannot be used (use --cache_aug_variants to cache augmented variants) / latentをキャッシュするときはcolor_augとrandom_cropは使えません（--cache_aug_variantsでaugmentation済みバリアントをキャッシュ可能です）"

        self.assert_extra_args(args, train_dataset_group, val_dataset_group)  # may change some args

        # acceleratorを準備する
        logger.info(f"preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        logger.info(f"prepared accelerator on {accelerator.device}")
        is_main_process = accelerator.is_main_process

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = (torch.float32 if args.no_half_vae else weight_dtype) if self.cast_vae(args) else None

        # load target models: unet may be None for lazy loading
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        if vae_dtype is None:
            vae_dtype = vae.dtype
            logger.info(f"vae_dtype is set to {vae_dtype} by the model since cast_vae() is false")

        if getattr(args, "vae_reflection_padding", False):
            vae = model_util.use_reflection_padding(vae)

        if args.use_ramtorch_vae:
            vae = apply_ramtorch_to_module(vae, "vae", accelerator.device, vae_dtype)

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # prepare dataset for latents caching if needed
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()

            train_dataset_group.new_cache_latents(vae, accelerator)
            if val_dataset_group is not None:
                val_dataset_group.new_cache_latents(vae, accelerator)

            # Initial in-memory variant generation (VAE is still on GPU)
            if aug_refresh_epochs > 0 and latents_aug_variants > 1:
                vae_encode_fn = lambda imgs: self.encode_images_to_latents(args, vae, imgs)
                train_dataset_group.refresh_latent_variants(vae_encode_fn, accelerator.device, vae_dtype)
                if val_dataset_group is not None:
                    val_dataset_group.refresh_latent_variants(vae_encode_fn, accelerator.device, vae_dtype)
                logger.info("Initial latent augmentation variants generated in-memory.")

            if self.adv_scale > 0.0:
                train_dataset_group.set_current_strategies()
                self.prepare_adv_reference(args, train_dataset_group, accelerator)

            vae.to("cpu")
            clean_memory_on_device(accelerator.device)

            accelerator.wait_for_everyone()

        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        text_encoding_strategy = self.get_text_encoding_strategy(args)
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        text_encoder_outputs_caching_strategy = self.get_text_encoder_outputs_caching_strategy(args)
        if text_encoder_outputs_caching_strategy is not None:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)
        self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, train_dataset_group, weight_dtype)
        if val_dataset_group is not None:
            self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, val_dataset_group, weight_dtype)

        # Initial in-memory caption TE variant generation.
        # TE models may have been moved back to CPU by cache_text_encoder_outputs_if_needed
        # (e.g. Anima's trainer). Save/restore device states so we don't change placement.
        if aug_refresh_epochs > 0 and caption_aug_variants > 1 and getattr(args, "cache_text_encoder_outputs", False):
            logger.info("Generating initial caption TE variants in-memory...")
            te_device_states = [(t_enc, next(t_enc.parameters()).device) for t_enc in text_encoders]
            for t_enc in text_encoders:
                t_enc.to(accelerator.device, dtype=weight_dtype)
            train_dataset_group.refresh_caption_te_variants(
                text_encoders, tokenize_strategy, text_encoding_strategy, accelerator
            )
            for t_enc, orig_device in te_device_states:
                t_enc.to(orig_device)
            clean_memory_on_device(accelerator.device)
            logger.info("Initial caption TE variants generated.")

        if unet is None:
            # lazy load unet if needed. text encoders may be freed or replaced with dummy models for saving memory
            unet, text_encoders = self.load_unet_lazily(args, weight_dtype, accelerator, text_encoders)

        # 差分追加学習のためにモデルを読み込む
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None and not isinstance(args.base_weights, list):
            args.base_weights = [args.base_weights]

        if args.base_weights_multiplier is not None and not isinstance(args.base_weights_multiplier, list):
            args.base_weights_multiplier = [float(x) for x in [args.base_weights_multiplier]]

        if args.base_weights is not None:
            # base_weights が指定されている場合は、指定された重みを読み込みマージする
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=", 1)
                net_kwargs[key] = value

        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            if "dropout" not in net_kwargs:
                # workaround for LyCORIS (;^ω^)
                net_kwargs["dropout"] = args.network_dropout

            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return

        # Store network reference for T-LoRA mask setter (LoCon flag path)
        self.network = network

        # GoRA: override args and net_kwargs to reflect enforced values in saved metadata
        _gora_algo = (net_kwargs.get("algo", "") or "").lower()
        if _gora_algo in ("gora", "ralora"):
            if args.network_alpha != args.network_dim:
                accelerator.print(
                    f"GoRA: overriding --network_alpha from {args.network_alpha} to {args.network_dim} "
                    f"(GoRA requires alpha = dim)"
                )
                args.network_alpha = args.network_dim

            if net_kwargs.get("use_scalar", "").lower() in ("true", "1", "yes"):
                accelerator.print(
                    "GoRA: overriding use_scalar from True to False "
                    "(scalar destabilizes importance convergence)"
                )
                net_kwargs["use_scalar"] = "False"

            conv_dim = net_kwargs.get("conv_dim", None)
            conv_alpha = net_kwargs.get("conv_alpha", None)
            if conv_dim is not None and conv_alpha is not None and int(conv_dim) > 0 and str(conv_alpha) != str(conv_dim):
                accelerator.print(
                    f"GoRA: overriding conv_alpha ({conv_alpha}) to conv_dim ({conv_dim}) "
                    f"(GoRA requires alpha = dim)"
                )
                net_kwargs["conv_alpha"] = str(conv_dim)

        network_has_multiplier = hasattr(network, "set_multiplier")

        # TODO remove `hasattr` by setting up methods if not defined in the network like below  (hacky but will work):
        # if not hasattr(network, "prepare_network"):
        #    network.prepare_network = lambda args: None

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        self.post_process_network(args, accelerator, network, text_encoders, unet)

        # apply network to unet and text_encoder
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            # FIXME consider alpha of weights: this assumes that the alpha is not changed
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.use_ramtorch_network:
            #move all network weights to cpu first as base device
            network = network.to("cpu")
            logger.info("Applying RamTorch to network/lora.")
            network = apply_ramtorch_to_module(network, "network/lora", accelerator.device)
            # Make sure the rest of the network is moved to the accelerator.device
            network = network.to(accelerator.device)

        if args.gradient_checkpointing:
            if args.cpu_offload_checkpointing:
                unet.enable_gradient_checkpointing(cpu_offload=True)
            else:
                unet.enable_gradient_checkpointing()

            for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders)):
                if flag:
                    if t_enc.supports_gradient_checkpointing:
                        t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")

        (
            optimizer_name, 
            optimizer_args, 
            optimizer, 
            optimizer_train_fn, 
            optimizer_eval_fn, 
            lr_descriptions, 
            text_encoder_lr
         ) = train_util.prepare_optimizer(args, network, accelerator)

        # prepare dataloader
        # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
        # some strategies can be None
        train_dataset_group.set_current_strategies()
        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()

        # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

        train_dataloader_generator = torch.Generator(device="cpu")
        train_dataloader_generator_init_seed = train_dataloader_generator.initial_seed()

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
            pin_memory=args.pin_data_loader_memory or args.pin_memory,
            generator=train_dataloader_generator,
        )

        if val_dataset_group is not None:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset_group if val_dataset_group is not None else [],
                shuffle=False,
                batch_size=1,
                collate_fn=collator,
                num_workers=n_workers,
                persistent_workers=args.persistent_data_loader_workers,
                pin_memory=args.pin_data_loader_memory or args.pin_memory,
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
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet_weight_dtype = te_weight_dtype = weight_dtype
        # Experimental Feature: Put base model into fp8 to save vram
        if args.fp8_base or args.fp8_base_unet:
            assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0 / fp8を使う場合はtorch>=2.1.0が必要です。"
            assert (
                args.mixed_precision != "no"
            ), "fp8_base requires mixed precision='fp16' or 'bf16' / fp8を使う場合はmixed_precision='fp16'または'bf16'が必要です。"
            accelerator.print("enable fp8 training for U-Net.")
            unet_weight_dtype = torch.float8_e4m3fn

            if not args.fp8_base_unet:
                accelerator.print("enable fp8 training for Text Encoder.")
            te_weight_dtype = weight_dtype if args.fp8_base_unet else torch.float8_e4m3fn

            # unet.to(accelerator.device)  # this makes faster `to(dtype)` below, but consumes 23 GB VRAM
            # unet.to(dtype=unet_weight_dtype)  # without moving to gpu, this takes a lot of time and main memory

            # logger.info(f"set U-Net weight dtype to {unet_weight_dtype}, device to {accelerator.device}")
            # unet.to(accelerator.device, dtype=unet_weight_dtype)  # this seems to be safer than above
            logger.info(f"set U-Net weight dtype to {unet_weight_dtype}")
            if not args.keep_unet_dtype:
                unet.to(dtype=unet_weight_dtype)  # do not move to device because unet is not prepared by accelerator
            else:
                accelerator.print(f"keeping U-Net in its loaded dtype (skip fp8 cast)")

        unet.requires_grad_(False)
        if self.cast_unet(args) and not args.keep_unet_dtype:
            unet.to(dtype=unet_weight_dtype)
        for i, t_enc in enumerate(text_encoders):
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
            if t_enc.device.type != "cpu" and self.cast_text_encoder(args):
                t_enc.to(dtype=te_weight_dtype)

                # nn.Embedding not support FP8
                if te_weight_dtype != weight_dtype:
                    self.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)

        network_needs_init = ((hasattr(network, "prepare_gora") and hasattr(network, "_gora_needs_init") and network._gora_needs_init) or
            (hasattr(network, "_ralora_needs_init") and network._ralora_needs_init))

        if network_needs_init:
            # Detect pre-cached latents — if first batch has latents, VAE isn't needed
            has_latents = False
            for batch in train_dataloader:
                if isinstance(batch, dict) and "latents" in batch and batch["latents"] is not None:
                    has_latents = True
                break

            # Save original devices for restoration after gora or ralora
            vae_orig_device = next(vae.parameters()).device
            unet_orig_device = next(unet.parameters()).device
            te_orig_devices = [
                next(t_enc.parameters()).device for t_enc in text_encoders
            ]

            # Move base models to accelerator device for GoRA forward pass
            # (accelerator.prepare hasn't run yet; models must be on same device as batch data)
            if not has_latents:
                vae.requires_grad_(False)
                vae.eval()
                vae.to(accelerator.device, dtype=vae_dtype)
            unet.to(accelerator.device, dtype=unet_weight_dtype if self.cast_unet(args) else None)

            temp_text_encoders = [
                (t_enc.to(accelerator.device) if flag else t_enc)
                for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))
            ]
            if len(text_encoders) > 1:
                temp_text_encoder = temp_text_encoders
            else:
                temp_text_encoder = temp_text_encoders[0]

            if args.gradient_checkpointing:
                # according to TI example in Diffusers, train is required
                unet.train()
                for i, (t_enc, frag) in enumerate(zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))):
                    t_enc.train()

                    # set top parameter requires_grad = True for gradient checkpointing works
                    if frag:
                        self.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)
            else:
                unet.eval()
                for t_enc in temp_text_encoders:
                    t_enc.eval()

            del t_enc

            network.prepare_grad_etc(temp_text_encoder, unet)

            # Create noise_scheduler early for GoRA forward pass
            # (normally created later; stateless — safe to create here)
            network_init_noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        # GoRA: precompute gradients for new GoRA networks (no-op for others and resumption)
        if hasattr(network, "prepare_gora") and hasattr(network, "_gora_needs_init") and network._gora_needs_init:
            accelerator.print("GoRA: Pre-computing gradients for rank allocation and initialization...")

            # Extract GoRA parameters from network_args
            gora_kwargs = {}
            for key, value in net_kwargs.items():
                if key.startswith("gora_"):
                    gora_kwargs[key] = value

            max_steps = int(gora_kwargs.get("gora_steps", gora_kwargs.get("gora_max_steps", 64)))
            adaptive_n = gora_kwargs.get("gora_adaptive_n", "True").lower() in ("true", "1", "yes")
            adaptive_gamma = gora_kwargs.get("gora_adaptive_gamma", "False").lower() in ("true", "1", "yes")

            # Build forward function using the trainer's process_batch
            def gora_forward_fn(batch):
                return self.process_batch(
                    batch=batch,
                    text_encoders=text_encoders,
                    unet=unet,
                    network=network,
                    vae=vae,
                    noise_scheduler=network_init_noise_scheduler,
                    vae_dtype=vae_dtype,
                    weight_dtype=weight_dtype,
                    accelerator=accelerator,
                    args=args,
                    text_encoding_strategy=text_encoding_strategy,
                    tokenize_strategy=tokenize_strategy,
                    is_train=True,
                    train_text_encoder=train_text_encoder,
                    train_unet=train_unet,
                    edm2_model=None,
                )

            network.prepare_gora(
                dataloader=train_dataloader,
                forward_fn=gora_forward_fn,
                max_steps=max_steps,
                adaptive_n=adaptive_n,
                adaptive_gamma=adaptive_gamma,
            )
            accelerator.print("GoRA: Pre-computation complete.")

        # RaLoRA: precompute gradients for new RaLoRA networks (no-op for others and resumption)
        if hasattr(network, "_ralora_needs_init") and network._ralora_needs_init:
            accelerator.print("RaLoRA: Pre-computing gradients for rank and GID initialization...")

            # Import RaLoRAModule
            from lycoris.modules.locon import RaLoRAModule
            
            # Extract RaLoRA parameters from network_args
            ralora_kwargs = {}
            for key, value in net_kwargs.items():
                if key.startswith("ralora_"):
                    ralora_kwargs[key] = value

            max_steps = int(ralora_kwargs.get("ralora_max_steps", 64))
            n_max = int(ralora_kwargs.get("ralora_n_max", 32))
            pro_mode = ralora_kwargs.get("ralora_pro", "False").lower() in ("true", "1", "yes")
            erank_method = ralora_kwargs.get("ralora_erank_method", "entropy")
            svd_threshold = float(ralora_kwargs.get("ralora_svd_threshold", 0.0))
            cumulative_variance = float(ralora_kwargs.get("ralora_cumulative_variance", 0.0))

            # Build forward function using the trainer's process_batch
            def ralora_forward_fn(batch):
                return self.process_batch(
                    batch=batch,
                    text_encoders=text_encoders,
                    unet=unet,
                    network=network,
                    vae=vae,
                    noise_scheduler=network_init_noise_scheduler,
                    vae_dtype=vae_dtype,
                    weight_dtype=weight_dtype,
                    accelerator=accelerator,
                    args=args,
                    text_encoding_strategy=text_encoding_strategy,
                    tokenize_strategy=tokenize_strategy,
                    is_train=True,
                    train_text_encoder=train_text_encoder,
                    train_unet=train_unet,
                    edm2_model=None,
                )

            # Compute world_size and global_rank for distributed
            world_size = accelerator.num_processes if hasattr(accelerator, 'num_processes') else 1
            global_rank = accelerator.process_index if hasattr(accelerator, 'process_index') else 0

            RaLoRAModule.precompute_and_init(
                model=network,
                dataloader=train_dataloader,
                forward_fn=ralora_forward_fn,
                max_steps=max_steps,
                n_max=n_max,
                pro_mode=pro_mode,
                erank_method=erank_method,
                svd_threshold=svd_threshold,
                cumulative_variance=cumulative_variance,
                world_size=world_size,
                global_rank=global_rank,
                device=accelerator.device,
                save_dir=args.output_dir if hasattr(args, 'output_dir') else None,
            )
            accelerator.print("RaLoRA: Pre-computation complete.")

        if network_needs_init:
            # Restore initial dataloader seed
            train_dataloader_generator.manual_seed(train_dataloader_generator_init_seed)

            # Restore models to their original devices
            vae.to(vae_orig_device)
            unet.to(unet_orig_device)
            for t_enc, orig_dev in zip(text_encoders, te_orig_devices):
                t_enc.to(orig_dev)

            # Free GPU memory fragmentation from gora or RaLoRA precompute
            gc.collect()
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()
                accelerator.print("GPU cache cleared.")

        # acceleratorがなんかよろしくやってくれるらしい / accelerator will do something good
        if args.deepspeed:
            flags = self.get_text_encoders_train_flags(args, text_encoders)
            ds_model = deepspeed_utils.prepare_deepspeed_model(
                args,
                unet=unet if train_unet else None,
                text_encoder1=text_encoders[0] if flags[0] else None,
                text_encoder2=(text_encoders[1] if flags[1] else None) if len(text_encoders) > 1 else None,
                network=network,
            )
            ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                ds_model, optimizer, train_dataloader, lr_scheduler
            )
            training_model = ds_model
        else:
            if train_unet:
                # default implementation is:  unet = accelerator.prepare(unet)
                unet = self.prepare_unet_with_accelerator(args, accelerator, unet)  # accelerator does some magic here
            else:
                # move to device because unet is not prepared by accelerator
                unet.to(accelerator.device, dtype=unet_weight_dtype if self.cast_unet(args) else None)
            if train_text_encoder:
                text_encoders = [
                    (accelerator.prepare(t_enc) if flag else t_enc)
                    for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))
                ]
                if len(text_encoders) > 1:
                    text_encoder = text_encoders
                else:
                    text_encoder = text_encoders[0]
            else:
                pass  # if text_encoder is not trained, no need to prepare. and device and dtype are already set

            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                network, optimizer, train_dataloader, lr_scheduler
            )
            training_model = network

        if val_dataset_group is not None:
            val_dataloader = accelerator.prepare(val_dataloader)
            cyclic_val_dataloader = itertools.cycle(val_dataloader)
        else:
            val_dataloader, cyclic_val_dataloader = None, None

        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for i, (t_enc, frag) in enumerate(zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))):
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if frag:
                    self.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)

        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        del t_enc

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        # before resuming make hook for saving/loading to save/load the network weights only
        def save_model_hook(models, weights, output_dir):
            # pop weights of other models than network to save only network weights
            # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
            if accelerator.is_main_process or args.deepspeed:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)
                # print(f"save model hook: {len(weights)} weights will be saved")

            # save current epoch and step
            train_state_file = os.path.join(output_dir, "train_state.json")
            # +1 is needed because the state is saved before current_step is set from global_step
            logger.info(f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value+1}")
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

            # save adaptive timestep sampler state if enabled
            if self.adaptive_manager is not None:
                adaptive_state_file = os.path.join(output_dir, "adaptive_sampler_state.json")
                logger.info(f"save adaptive sampler state to {adaptive_state_file}")
                adaptive_state = self.adaptive_manager.state_dict()
                # Convert tensors to lists for JSON serialization
                adaptive_state_serializable = {
                    "sampler_network": {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in adaptive_state["sampler_network"].items()},
                    "optimizer": adaptive_state["optimizer"],
                    "queue": [q.tolist() if isinstance(q, torch.Tensor) else q for q in adaptive_state["queue"]],
                    "learning_rate": adaptive_state["learning_rate"],
                    "entropy_coeff": adaptive_state["entropy_coeff"],
                    "f_s": adaptive_state["f_s"],
                    "queue_size": adaptive_state["queue_size"],
                    "num_selected": adaptive_state["num_selected"],
                    "v_parameterization": adaptive_state.get("v_parameterization", False),
                }
                with open(adaptive_state_file, "w", encoding="utf-8") as f:
                    json.dump(adaptive_state_serializable, f)

            # save QMC timestep sampling sequence position if enabled, so the
            # low-discrepancy coverage is preserved across checkpoint resume.
            _qmc_method = getattr(args, "qmc_timestep_sampling", None)
            if _qmc_method is not None:
                _qmc_seed = getattr(args, "qmc_seed", 0)
                _qmc_rank = accelerator.process_index if hasattr(accelerator, "process_index") else 0
                _qmc_mgr = _QMCSequenceManager(method=_qmc_method, seed=_qmc_seed, rank=_qmc_rank)
                qmc_state_file = os.path.join(output_dir, "qmc_state.json")
                logger.info(f"save QMC sequence state to {qmc_state_file}")
                with open(qmc_state_file, "w", encoding="utf-8") as f:
                    json.dump(_qmc_mgr.state_dict(), f)

        steps_from_state = None

        def load_model_hook(models, input_dir):
            # remove models except network
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)
            # print(f"load model hook: {len(models)} models will be loaded")

            # load current epoch and step to
            nonlocal steps_from_state
            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                steps_from_state = data["current_step"]
                logger.info(f"load train state from {train_state_file}: {data}")

            # load adaptive timestep sampler state if available
            if self.adaptive_manager is not None:
                adaptive_state_file = os.path.join(input_dir, "adaptive_sampler_state.json")
                if os.path.exists(adaptive_state_file):
                    logger.info(f"load adaptive sampler state from {adaptive_state_file}")
                    with open(adaptive_state_file, "r", encoding="utf-8") as f:
                        adaptive_state_serializable = json.load(f)
                    # Convert lists back to tensors
                    adaptive_state = {
                        "sampler_network": {k: torch.tensor(v, device=accelerator.device) if isinstance(v, list) else v for k, v in adaptive_state_serializable["sampler_network"].items()},
                        "optimizer": adaptive_state_serializable["optimizer"],
                        "queue": [torch.tensor(q, device=accelerator.device) if isinstance(q, list) else q for q in adaptive_state_serializable["queue"]],
                        "learning_rate": adaptive_state_serializable["learning_rate"],
                        "entropy_coeff": adaptive_state_serializable["entropy_coeff"],
                        "f_s": adaptive_state_serializable["f_s"],
                        "queue_size": adaptive_state_serializable["queue_size"],
                        "num_selected": adaptive_state_serializable["num_selected"],
                    }
                    self.adaptive_manager.load_state_dict(adaptive_state)

            # load QMC timestep sampling sequence position if available, so the
            # low-discrepancy coverage continues from where it left off.
            _qmc_method = getattr(args, "qmc_timestep_sampling", None)
            if _qmc_method is not None:
                qmc_state_file = os.path.join(input_dir, "qmc_state.json")
                if os.path.exists(qmc_state_file):
                    logger.info(f"load QMC sequence state from {qmc_state_file}")
                    with open(qmc_state_file, "r", encoding="utf-8") as f:
                        qmc_state = json.load(f)
                    _qmc_seed = getattr(args, "qmc_seed", 0)
                    _qmc_rank = accelerator.process_index if hasattr(accelerator, "process_index") else 0
                    _qmc_mgr = _QMCSequenceManager(method=_qmc_method, seed=_qmc_seed, rank=_qmc_rank)
                    _qmc_mgr.load_state_dict(qmc_state)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # resumeする
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        accelerator.print(
            f"  num validation images * repeats / 学習画像の数×繰り返し回数: {val_dataset_group.num_train_images if val_dataset_group is not None else 0}"
        )
        accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_validation_images": val_dataset_group.num_train_images if val_dataset_group is not None else 0,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_adaptive_timestep_sampling": bool(getattr(args, "adaptive_timestep_sampling", False)),
            "ss_adaptive_sampler_lr": args.adaptive_sampler_lr if getattr(args, "adaptive_timestep_sampling", False) else None,
            "ss_adaptive_sampler_entropy_coeff": args.adaptive_sampler_entropy_coeff if getattr(args, "adaptive_timestep_sampling", False) else None,
            "ss_adaptive_sampler_update_freq": args.adaptive_sampler_update_freq if getattr(args, "adaptive_timestep_sampling", False) else None,
            "ss_adaptive_sampler_eval_chunk_size": getattr(args, "adaptive_sampler_eval_chunk_size", 16) if getattr(args, "adaptive_timestep_sampling", False) else None,
            "ss_adaptive_sampler_eval_stride": getattr(args, "adaptive_sampler_eval_stride", 1) if getattr(args, "adaptive_timestep_sampling", False) else None,
            "ss_adaptive_sampler_fp32_eval": bool(getattr(args, "adaptive_sampler_fp32_eval", False)) if getattr(args, "adaptive_timestep_sampling", False) else None,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
            "ss_debiased_estimation": bool(args.debiased_estimation_loss),
            "ss_noise_offset_random_strength": args.noise_offset_random_strength,
            "ss_ip_noise_gamma_random_strength": args.ip_noise_gamma_random_strength,
            "ss_loss_type": args.loss_type,
            "ss_huber_schedule": args.huber_schedule,
            "ss_huber_scale": args.huber_scale,
            "ss_huber_c": args.huber_c,
            "ss_fp8_base": bool(args.fp8_base),
            "ss_fp8_base_unet": bool(args.fp8_base_unet),
            "ss_validation_seed": args.validation_seed,
            "ss_validation_split": float(args.validation_split),
            "ss_max_validation_steps": args.max_validation_steps,
            "ss_validate_every_n_epochs": args.validate_every_n_epochs,
            "ss_validate_every_n_steps": args.validate_every_n_steps,
            "ss_resize_interpolation": args.resize_interpolation,
            "ss_focal_frequency_loss": bool(getattr(args, "focal_frequency_loss", False)),
            "ss_focal_frequency_loss_weight": getattr(args, "focal_frequency_loss_weight", 1.0),
            "ss_focal_frequency_loss_alpha": getattr(args, "focal_frequency_loss_alpha", 1.0),
            "ss_patch_topology_loss": bool(getattr(args, "patch_topology_loss", False)),
            "ss_patch_topology_weight": getattr(args, "patch_topology_weight", 1.0),
            "ss_patch_topology_tau": getattr(args, "patch_topology_tau", 0.1),
            "ss_patch_topology_scale_levels": getattr(args, "patch_topology_scale_levels", 2),
            "ss_patch_topology_loss_type": getattr(args, "patch_topology_loss_type", "kl"),
            "ss_patch_topology_disable_timestep_weight": bool(getattr(args, "patch_topology_disable_timestep_weight", False)),
            "ss_patch_topology_chunk_size": getattr(args, "patch_topology_chunk_size", 512),
            "ss_patch_topology_start_step": getattr(args, "patch_topology_start_step", 0),
            "ss_patch_topology_warmup_steps": getattr(args, "patch_topology_warmup_steps", 0),
            "ss_patch_topology_dynamic_weighting": getattr(args, "patch_topology_dynamic_weighting", "none"),
            "ss_patch_topology_dwa_temperature": getattr(args, "patch_topology_dwa_temperature", 2.0),
            "ss_patch_topology_gradnorm_alpha": getattr(args, "patch_topology_gradnorm_alpha", 1.5),
            "ss_patch_topology_dynamic_max_weight": getattr(args, "patch_topology_dynamic_max_weight", 10.0),
            "ss_hf_scale": self.hf_scale if self.hf_scale > 0.0 else 0.0,
            "ss_hf_exponent": self.hf_exponent if self.hf_scale > 0.0 else None,
            "ss_hf_patch": self.hf_patch if self.hf_scale > 0.0 else None,
            "ss_hf_high_noise_snr_cut": self.hf_high_noise_snr_cut if self.hf_scale > 0.0 else None,
            "ss_hf_high_noise_min_weight": self.hf_high_noise_min_weight if self.hf_scale > 0.0 else None,
            "ss_hf_high_noise_power": self.hf_high_noise_power if self.hf_scale > 0.0 else None,
            "ss_anchor_scale": self.anchor_scale if self.anchor_scale > 0.0 else 0.0,
            "ss_anchor_levels": self.anchor_levels if self.anchor_scale > 0.0 else None,
            "ss_anchor_snr_weighting": bool(self.anchor_snr_weighting) if self.anchor_scale > 0.0 else None,
            "ss_adv_scale": self.adv_scale if self.adv_scale > 0.0 else 0.0,
            "ss_adv_levels": self.adv_levels if self.adv_scale > 0.0 else None,
            "ss_adv_domain": self.adv_domain if self.adv_scale > 0.0 else None,
            "ss_adv_features": self.adv_features if self.adv_scale > 0.0 else None,
            "ss_adv_bandwidths": self.adv_bandwidths if self.adv_scale > 0.0 else None,
            "ss_adv_tokens_per_band": self.adv_tokens_per_band if self.adv_scale > 0.0 else None,
            "ss_adv_reference_samples": self.adv_reference_samples if self.adv_scale > 0.0 else None,
            "ss_adv_rff_seed": self.adv_rff_seed if self.adv_scale > 0.0 else None,
            "ss_adv_snr_weighting": bool(self.adv_snr_weighting) if self.adv_scale > 0.0 else None,
        }

        self.update_metadata(metadata, args)  # architecture specific metadata

        if use_user_config:
            # save metadata of multiple datasets
            # NOTE: pack "ss_datasets" value as json one time
            #   or should also pack nested collections as json?
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor

            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "skip_image_resolution": dataset.skip_image_resolution,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                    "resize_interpolation": dataset.resize_interpolation,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "random_crop_padding_percent": float(getattr(subset, "random_crop_padding_percent", 0.05)),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                        "keep_tokens_separator": subset.keep_tokens_separator,
                        "secondary_separator": subset.secondary_separator,
                        "enable_wildcard": bool(subset.enable_wildcard),
                        "caption_prefix": subset.caption_prefix,
                        "caption_suffix": subset.caption_suffix,
                        "resize_interpolation": subset.resize_interpolation,
                        "resolution": subset.resolution,
                        "min_bucket_reso": subset.min_bucket_reso,
                        "max_bucket_reso": subset.max_bucket_reso,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        subset_metadata["is_val"] = subset.is_val
                        if subset.is_reg or subset.is_val:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                    # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                    # なので、ここで複数datasetの回数を合算してもあまり意味はない
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            val_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    if subset.is_reg:
                        info = reg_dataset_dirs_info
                    elif subset.is_val:
                        info = val_dataset_dirs_info
                    else:
                        info = dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            metadata.update(
                {
                    "ss_batch_size_per_device": args.train_batch_size,
                    "ss_total_batch_size": total_batch_size,
                    "ss_resolution": args.resolution,
                    "ss_color_aug": bool(args.color_aug),
                    "ss_flip_aug": bool(args.flip_aug),
                    "ss_random_crop": bool(args.random_crop),
                    "ss_random_crop_padding_percent": float(getattr(args, "random_crop_padding_percent", 0.05)),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_multires_training": bool(getattr(dataset, "multires_training", False)),
                    "ss_resolution_jitter": json.dumps(
                        {
                            "resolutions": dataset.resolution_jitter_resolutions,
                            "batch_sizes": dataset.resolution_jitter_batch_sizes,
                            "weights": dataset.resolution_jitter_weights,
                        }
                    )
                    if getattr(dataset, "has_resolution_jitter", False)
                    else None,
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_skip_image_resolution": dataset.skip_image_resolution,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                    "ss_bucket_info": json.dumps(dataset.bucket_info),
                }
            )

        # add extra args
        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata["ss_vae_scale_factor"] = self.vae_scale_factor
        metadata["ss_vae_shift_factor"] = self.latent_shift
        metadata["ss_vae_reflection_padding"] = getattr(args, "vae_reflection_padding", False)

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        # calculate steps to skip when resuming or starting from a specific step
        initial_step = 0
        if args.initial_epoch is not None or args.initial_step is not None:
            # if initial_epoch or initial_step is specified, steps_from_state is ignored even when resuming
            if steps_from_state is not None:
                logger.warning(
                    "steps from the state is ignored because initial_step is specified / initial_stepが指定されているため、stateからのステップ数は無視されます"
                )
            if args.initial_step is not None:
                initial_step = args.initial_step
            else:
                # num steps per epoch is calculated by gradient_accumulation_steps (dataloader len is already divided by num_processes)
                initial_step = (args.initial_epoch - 1) * math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
        else:
            # if initial_epoch and initial_step are not specified, steps_from_state is used when resuming
            if steps_from_state is not None:
                initial_step = steps_from_state
                steps_from_state = None

        if initial_step > 0:
            assert (
                args.max_train_steps > initial_step
            ), f"max_train_steps should be greater than initial step / max_train_stepsは初期ステップより大きい必要があります: {args.max_train_steps} vs {initial_step}"

        resumed_step = initial_step
        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
                if not args.resume:
                    logger.info(
                        f"initial_step is specified but not resuming. lr scheduler will be started from the beginning / initial_stepが指定されていますがresumeしていないため、lr schedulerは最初から始まります"
                    )
                update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                epoch_to_start = resumed_step // update_steps_per_epoch
                # Calculate exactly how many batches to skip in the starting epoch
                initial_step = (resumed_step % update_steps_per_epoch) * args.gradient_accumulation_steps
                logger.info(f"skipping {epoch_to_start} epochs and {initial_step} batches / {epoch_to_start}エポックと{initial_step}バッチをスキップします")
            else:
                # if not, only epoch no is skipped for informative purpose
                epoch_to_start = resumed_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                initial_step = 0  # do not skip

        global_step = resumed_step

        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        # Initialize T-LoRA timestep masking if applicable
        self.setup_tlora_masking(net_kwargs, args.network_dim, noise_scheduler)

        # Initialize LoRA² rank regularization if applicable
        self.setup_lora2_regularization(net_kwargs)

        # Initialize Focal Frequency Loss if enabled
        if getattr(args, "focal_frequency_loss", False):
            self.ffl_enabled = True
            self.ffl_module = FocalFrequencyLoss(
                loss_weight=float(args.focal_frequency_loss_weight),
                alpha=float(args.focal_frequency_loss_alpha),
            )
            logger.info(
                f"Focal Frequency Loss enabled: weight={args.focal_frequency_loss_weight}, "
                f"alpha={args.focal_frequency_loss_alpha}"
            )

        # Initialize Patch Topology Loss if enabled
        if getattr(args, "patch_topology_loss", False):
            self.patch_topology_enabled = True
            self.patch_topology_full_weight = float(getattr(args, "patch_topology_weight", 1.0))
            self.patch_topology_start_step = int(getattr(args, "patch_topology_start_step", 0))
            self.patch_topology_warmup_steps = int(getattr(args, "patch_topology_warmup_steps", 0))
            self.patch_topology_loss_module = PatchTopologyLoss(
                loss_weight=1.0,  # effective weight applied dynamically via warmup/start_step
                tau_latent=float(getattr(args, "patch_topology_tau", 0.1)),
                tau_target=float(getattr(args, "patch_topology_tau", 0.1)),
                scale_levels=int(getattr(args, "patch_topology_scale_levels", 2)),
                loss_type=getattr(args, "patch_topology_loss_type", "kl"),
                apply_timestep_weight=not getattr(args, "patch_topology_disable_timestep_weight", False),
                chunk_size=int(getattr(args, "patch_topology_chunk_size", 512)),
            )
            self.patch_topology_loss_module.to(accelerator.device)
            self.patch_topology_weighter = build_weighter_from_args(args, self.patch_topology_full_weight)
            logger.info(
                f"Patch Topology Loss enabled: weight={self.patch_topology_full_weight}, "
                f"tau={getattr(args, 'patch_topology_tau', 0.1)}, "
                f"scale_levels={getattr(args, 'patch_topology_scale_levels', 2)}, "
                f"loss_type={getattr(args, 'patch_topology_loss_type', 'kl')}, "
                f"timestep_weight={not getattr(args, 'patch_topology_disable_timestep_weight', False)}, "
                f"chunk_size={getattr(args, 'patch_topology_chunk_size', 512)}, "
                f"start_step={self.patch_topology_start_step}, "
                f"warmup_steps={self.patch_topology_warmup_steps}, "
                f"dynamic_weighting={getattr(args, 'patch_topology_dynamic_weighting', 'none')}"
            )

        # Initialize Latent Wavelet Diffusion (LWD) masking if enabled
        if getattr(args, "wavelet_masking", False):
            self.wavelet_masking_enabled = True
            self.wavelet_dwt = train_util.setup_wavelet_dwt(accelerator.device)
            logger.info(
                f"Latent Wavelet Diffusion (LWD) masking enabled: "
                f"l_bound={getattr(args, 'wavelet_mask_l_bound', 0.3)}, "
                f"wavelet=haar, J=1"
            )

        # Initialize Weight Noising if the network supports it
        _wn_sigma = net_kwargs.get("weight_noise_sigma", None)
        if _wn_sigma is not None and float(_wn_sigma) > 0:
            self.weight_noise_enabled = True
            _wn_mode = net_kwargs.get("weight_noise_mode", "relative")
            logger.info(
                f"Weight noising enabled: sigma={_wn_sigma}, mode={_wn_mode}"
            )

        edm2_model, edm2_optimizer, edm2_lr_scheduler = prepare_edm2_loss_weighting(args, noise_scheduler, accelerator)

        # Initialize Adaptive Timestep Sampler
        if getattr(args, "adaptive_timestep_sampling", False):
            # Adaptive sampling produces its own timesteps (passed as fixed_timesteps),
            # which bypasses the antithetic/stratified/QMC variance-reduction paths. Warn
            # so the user knows they are mutually exclusive (adaptive takes precedence).
            if (
                getattr(args, "antithetic_timestep_sampling", False)
                or getattr(args, "stratified_timestep_sampling", False)
                or getattr(args, "qmc_timestep_sampling", None) is not None
            ):
                logger.warning(
                    "Both --adaptive_timestep_sampling and "
                    "--antithetic_timestep_sampling/--stratified_timestep_sampling/--qmc_timestep_sampling "
                    "are enabled. Adaptive timestep sampling takes precedence and supplies fixed "
                    "timesteps, so antithetic/stratified/QMC variance reduction will NOT be applied. "
                    "Disable one of them to avoid this conflict."
                )
            adaptive_model_type = self.get_adaptive_model_type(args)
            adaptive_min_ts = 0 if args.min_timestep is None else args.min_timestep
            adaptive_max_ts = args.max_timestep  # None → defaults to num_train_timesteps in manager
            self._adaptive_disable_empty_cache = getattr(args, "adaptive_sampler_disable_empty_cache", False)
            self.adaptive_manager = AdaptiveTimestepManager(
                # Network is lazily initialized on first sample_timesteps() call,
                # inferring in_channels from the actual latent tensor shape.
                # This correctly handles any VAE (4-ch SD1.5/SDXL, 16-ch Flux/SD3/Anima).
                noise_scheduler=noise_scheduler,
                device=accelerator.device,
                dtype=weight_dtype,
                learning_rate=args.adaptive_sampler_lr,
                entropy_coeff=args.adaptive_sampler_entropy_coeff,
                update_freq=args.adaptive_sampler_update_freq,
                queue_size=args.adaptive_sampler_queue_size,
                num_selected=args.adaptive_sampler_num_selected,
                v_parameterization=args.v_parameterization,
                model_type=adaptive_model_type,
                hidden_channels=args.adaptive_sampler_hidden_channels,
                hidden_depth=args.adaptive_sampler_hidden_depth,
                min_timestep=adaptive_min_ts,
                max_timestep=adaptive_max_ts,
                eval_chunk_size=getattr(args, "adaptive_sampler_eval_chunk_size", 16),
                eval_stride=getattr(args, "adaptive_sampler_eval_stride", 1),
                fp32_eval=getattr(args, "adaptive_sampler_fp32_eval", False),
            )
            logger.info(f"Adaptive non-uniform timestep sampling enabled (model_type={adaptive_model_type})")

        # Warn about variance-reduction pairing breakage under gradient accumulation or
        # DDP. Antithetic assumes each (u, 1-u) pair lives in the same loss/gradient
        # aggregation unit; with gradient accumulation the batch is split into
        # micro-batches and with DDP it is sharded across ranks, so pairs may be
        # separated, reducing the variance-reduction benefit. (QMC is DDP-safe: each
        # rank offsets its scramble seed by its process index, so ranks draw from
        # different scrambled low-discrepancy sequences with no duplicate points.)
        if (
            getattr(args, "antithetic_timestep_sampling", False)
            or getattr(args, "stratified_timestep_sampling", False)
            or getattr(args, "qmc_timestep_sampling", None) is not None
        ):
            _ga = getattr(args, "gradient_accumulation_steps", 1)
            _np = accelerator.num_processes if hasattr(accelerator, "num_processes") else 1
            if _ga > 1 or _np > 1:
                logger.warning(
                    f"Antithetic/stratified/QMC timestep sampling is enabled with "
                    f"gradient_accumulation_steps={_ga} and num_processes={_np}. "
                    "Antithetic mirrored pairs may be split across micro-batches or GPU "
                    "ranks, which reduces (and can eliminate) the variance-reduction "
                    "benefit of antithetic. (QMC is DDP-safe: each rank consumes a disjoint "
                    "slice of the global low-discrepancy sequence.) For full antithetic "
                    "benefit, use a single GPU with gradient_accumulation_steps=1, or form "
                    "pairs within each micro-batch/rank."
                )

        train_util.init_trackers(accelerator, args, "network_train")

        loss_recorder = train_util.EMARecorder()
        val_loss_recorder = train_util.EMARecorder()
        rate_tracker = train_util.RateTracker()

        if args.edm2_loss_weighting:
            loss_scaled_recorder = train_util.EMARecorder()
            loss_edm2_recorder = train_util.EMARecorder()

        # NOTE: train_dataset_group is intentionally NOT deleted here.
        # The DataLoader holds a reference regardless (so `del` never freed
        # the object), and the epoch-variant refresh path
        # (cache_aug_refresh_epochs) needs the name to call
        # refresh_latent_variants / refresh_caption_te_variants.
        if val_dataset_group is not None:
            del val_dataset_group

        # callback for step start
        if hasattr(accelerator.unwrap_model(network), "on_step_start"):
            on_step_start_for_network = accelerator.unwrap_model(network).on_step_start
        else:
            on_step_start_for_network = lambda *args, **kwargs: None

        # function for saving/removing
        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False, dtype_override=None):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = self.get_sai_model_spec(args)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, dtype_override or save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # if text_encoder is not needed for training, delete it to save memory.
        # TODO this can be automated after SDXL sample prompt cache is implemented
        # Keep TE models alive when caption-variant epoch refresh is active,
        # because refresh_caption_te_variants() needs them on GPU each epoch.
        _need_te_for_refresh = (
            aug_refresh_epochs > 0
            and caption_aug_variants > 1
            and getattr(args, "cache_text_encoder_outputs", False)
        )
        if self.is_text_encoder_not_needed_for_training(args) and not _need_te_for_refresh:
            logger.info("text_encoder is not needed for training. deleting to save memory.")
            for t_enc in text_encoders:
                del t_enc
            text_encoders = []
            text_encoder = None
            gc.collect()
            clean_memory_on_device(accelerator.device)

        current_val_loss, average_val_loss, val_logs = None, None, {}
        keys_scaled, mean_norm, maximum_norm = None, None, None
        mean_grad_norm, mean_combined_norm = None, None
        max_mean_logs = {}
        current_global_step_loss = 0.0
        current_global_step_loss_scaled = 0.0 if args.edm2_loss_weighting else None
        average_loss_scaled = 0.0 if args.edm2_loss_weighting else None
        current_global_step_loss_edm2 = 0.0 if args.edm2_loss_weighting else None
        average_loss_edm2 = 0.0 if args.edm2_loss_weighting else None
        current_global_step_ffl = 0.0 if self.ffl_enabled else None
        current_global_step_patch_topo = 0.0 if self.patch_topology_enabled else None
        current_global_step_wav_mask = 0.0 if self.wavelet_masking_enabled else None
        current_global_step_wnoise = 0.0 if self.weight_noise_enabled else None
        current_global_step_hf = 0.0 if self.hf_scale > 0.0 else None
        current_global_step_anchor = 0.0 if self.anchor_scale > 0.0 else None
        current_global_step_adv = 0.0 if self.adv_scale > 0.0 else None
        avr_loss = 0.0
        accumulation_counter = 0
        accumulated_samples = 0  # Tracks actual samples across micro-batches for dynamic sigma

        # Detect step_func optimizer (e.g. AdamWScheduleFreePlus with Polyak step size).
        # step_func(function_value) replaces step() and requires the current loss value.
        # After accelerator.prepare(), the optimizer is wrapped in AcceleratedOptimizer
        # which doesn't expose step_func. The raw optimizer lives at optimizer.optimizer.
        _raw_optimizer = getattr(optimizer, 'optimizer', optimizer)
        _use_step_func = hasattr(_raw_optimizer, 'step_func') and callable(getattr(_raw_optimizer, 'step_func'))
        _step_func_loss_accum = 0.0

        # For --sample_at_first
        if train_util.sample_images_check(args, 0, global_step) or train_util.calculate_val_loss_check(args, global_step, 0, val_dataloader, train_dataloader):
            with torch.no_grad():
                #Switch network to eval mode
                accelerator.unwrap_model(network).eval()
                if args.gradient_checkpointing:
                    accelerator.unwrap_model(unet).eval()
                    for t_enc in text_encoders:
                        accelerator.unwrap_model(t_enc).eval()

                optimizer_eval_fn()
                self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
                if train_util.calculate_val_loss_check(args, global_step, 0, val_dataloader, train_dataloader):
                    current_val_loss, average_val_loss, val_logs = self.calculate_val_loss(
                        global_step, 0, train_dataloader, val_loss_recorder, val_dataloader, 
                        cyclic_val_dataloader, network, tokenize_strategy, 
                        text_encoders, text_encoding_strategy, unet, vae, noise_scheduler, 
                        vae_dtype, weight_dtype, accelerator, args, 0, None, train_text_encoder)
                #Switch network to train mode
                optimizer_train_fn()
                accelerator.unwrap_model(network).train()
                if args.gradient_checkpointing:
                    accelerator.unwrap_model(unet).train()
                    for t_enc in text_encoders:
                        accelerator.unwrap_model(t_enc).train()

        if plot_edm2_loss_weighting_check(args, global_step):
            plot_edm2_loss_weighting(args, global_step, edm2_model, noise_scheduler.config.num_train_timesteps, accelerator.device)

        is_tracking = len(accelerator.trackers) > 0
        if is_tracking:
            logs = self.generate_step_logs(
                args,
                current_global_step_loss,
                avr_loss,
                lr_scheduler,
                lr_descriptions,
                optimizer,
                keys_scaled,
                mean_norm,
                maximum_norm,
                mean_grad_norm,
                mean_combined_norm,
                edm2_lr_scheduler,
                current_global_step_loss_scaled,
                average_loss_scaled,
                current_global_step_loss_edm2,
                average_loss_edm2,
                current_val_loss=current_val_loss,
                average_val_loss=average_val_loss,
                current_ffl_loss=None,
                current_patch_topology_loss=None,
                current_hf_loss=None,
                current_anchor_loss=None,
                current_adv_loss=None,
                it_s=rate_tracker.it_per_sec,
            )
            if val_logs:
                logs.update(val_logs)
            # log empty object to commit the sample images to wandb
            accelerator.log(logs, step=0)

        # training loop
        if initial_step > 0:  # only if skip_until_initial_step is specified
            logger.info(f"skipping {initial_step} batches in the first epoch / 最初の{epoch_to_start}エポック内で{initial_step}バッチをスキップします")

        # log device and dtype for each model
        logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
        for i, t_enc in enumerate(text_encoders):
            params_itr = t_enc.parameters()
            params_itr.__next__()  # skip the first parameter
            params_itr.__next__()  # skip the second parameter. because CLIP first two parameters are embeddings
            param_3rd = params_itr.__next__()
            logger.info(f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}")

        clean_memory_on_device(accelerator.device)

        progress_bar = tqdm(
            range(args.max_train_steps - global_step), smoothing=0.1, disable=not accelerator.is_local_main_process, desc="steps",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
        )

        for epoch in range(epoch_to_start, num_train_epochs):
            current_epoch.value = epoch + 1
            accelerator.print(f"\nepoch {current_epoch.value}/{num_train_epochs}\n")

            metadata["ss_epoch"] = str(current_epoch.value)

            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)  # network.train() is called here

            # Epoch-variant refresh: regenerate augmentation variants in-memory
            if (aug_refresh_epochs > 0
                and epoch > epoch_to_start
                and (epoch - epoch_to_start) % aug_refresh_epochs == 0):
                # Refresh latent variants (requires VAE on GPU)
                if latents_aug_variants > 1 and cache_latents:
                    logger.info(f"Refreshing latent augmentation variants for epoch {current_epoch.value}...")
                    vae.to(accelerator.device, dtype=vae_dtype)
                    vae_encode_fn = lambda imgs: self.encode_images_to_latents(args, vae, imgs)
                    train_dataset_group.refresh_latent_variants(vae_encode_fn, accelerator.device, vae_dtype)
                    vae.to("cpu")
                    clean_memory_on_device(accelerator.device)
                    logger.info("Latent augmentation variants refreshed.")
                # Refresh caption TE variants (requires TE models on GPU for encoding)
                if caption_aug_variants > 1 and getattr(args, "cache_text_encoder_outputs", False):
                    logger.info(f"Refreshing caption TE variants for epoch {current_epoch.value}...")
                    # Save TE device states and move to GPU for encoding
                    te_device_states = [(t_enc, next(t_enc.parameters()).device) for t_enc in text_encoders]
                    for t_enc in text_encoders:
                        t_enc.to(accelerator.device, dtype=weight_dtype)
                    train_dataset_group.refresh_caption_te_variants(
                        text_encoders, tokenize_strategy, text_encoding_strategy, accelerator
                    )
                    # Restore TEs to their original devices (CPU for archs that don't train TEs)
                    for t_enc, orig_device in te_device_states:
                        t_enc.to(orig_device)
                    clean_memory_on_device(accelerator.device)
                    logger.info("Caption TE variants refreshed.")

                # Synchronize all processes after refresh
                accelerator.wait_for_everyone()

            # TRAINING
            skipped_dataloader = None
            if initial_step > 0:
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, initial_step - 1)
                initial_step = 1

            for step, batch in enumerate(skipped_dataloader or train_dataloader):
                current_step.value = global_step

                if initial_step > 0:
                    initial_step -= 1
                    continue

                # Set adaptive update flag: stash tensors only on steps where Algorithm 2 will run
                if self.adaptive_manager is not None:
                    self._adaptive_update_pending = self.adaptive_manager.should_update(global_step)

                with train_util.determine_grad_sync_context(args, accelerator, None, training_model, edm2_model):
                    on_step_start_for_network(text_encoder, unet)

                    accumulation_counter += 1

                    # Track actual micro-batch size for accurate effective batch size
                    _actual_bs = None
                    if "latents" in batch and batch["latents"] is not None:
                        _actual_bs = batch["latents"].shape[0]
                    elif "images" in batch and batch["images"] is not None:
                        _actual_bs = batch["images"].shape[0]
                    elif "captions" in batch:
                        _actual_bs = len(batch["captions"])
                    if _actual_bs is None:
                        _actual_bs = args.train_batch_size
                    accumulated_samples += _actual_bs

                    # preprocess batch for each model
                    self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True)

                    # Update patch topology current step for warmup/start_step logic
                    self._patch_topology_current_step = global_step

                    loss, pre_scaling_loss, loss_scaled = self.process_batch(
                        batch,
                        text_encoders,
                        unet,
                        network,
                        vae,
                        noise_scheduler,
                        vae_dtype,
                        weight_dtype,
                        accelerator,
                        args,
                        text_encoding_strategy,
                        tokenize_strategy,
                        is_train=True,
                        train_text_encoder=train_text_encoder,
                        train_unet=train_unet,
                        edm2_model=edm2_model,
                    )

                    # Track FFL loss for logging
                    if self.ffl_enabled and self.ffl_loss_value is not None:
                        current_global_step_ffl = (current_global_step_ffl or 0.0) + self.ffl_loss_value

                    # Track Patch Topology loss for logging
                    if self.patch_topology_enabled and self.patch_topology_loss_value is not None:
                        current_global_step_patch_topo = (current_global_step_patch_topo or 0.0) + self.patch_topology_loss_value

                    # Track wavelet mask ratio for logging
                    if self.wavelet_masking_enabled:
                        current_global_step_wav_mask = (current_global_step_wav_mask or 0.0) + self._wavelet_mask_ratio

                    # Track High-Frequency Token loss for logging (materialized at the
                    # existing periodic sync — the hot path keeps a detached tensor)
                    if self.hf_scale > 0.0 and self.hf_loss_value is not None:
                        current_global_step_hf = (current_global_step_hf or 0.0) + self.hf_loss_value.item()

                    # Track multiscale anchor loss for logging (materialized at the
                    # existing periodic sync — the hot path keeps a detached tensor)
                    if self.anchor_scale > 0.0 and self.anchor_loss_value is not None:
                        current_global_step_anchor = (current_global_step_anchor or 0.0) + self.anchor_loss_value.item()

                    if self.adv_scale > 0.0 and self.adv_loss_value is not None:
                        current_global_step_adv = (current_global_step_adv or 0.0) + self.adv_loss_value.item()

                    if loss.ndim != 0:
                        loss = loss.mean()

                    accelerator.backward(loss)

                    if self.should_sync_ramtorch(args, accelerator):
                        torch.cuda.synchronize()

                    edm2_loss = loss
                    loss = pre_scaling_loss

                    # Accumulate loss for step_func optimizer (Polyak step size needs function value)
                    # Adaptive timestep sampling: compute per-timestep losses with theta_k (before optimizer step)
                    if self.adaptive_manager is not None and accelerator.sync_gradients:
                        self.compute_adaptive_delta_before_step(unet, noise_scheduler, weight_dtype, accelerator, global_step)

                    if _use_step_func:
                        _step_func_loss_accum += loss.detach().item()

                    if accelerator.sync_gradients:
                        self.all_reduce_network(accelerator, network)  # sync DDP grad manually

                        # Sync and clip EDM2 gradients
                        if args.edm2_loss_weighting:
                            self.all_reduce_edm2_model(accelerator, edm2_model)

                        if args.max_grad_norm != 0.0 or args.edm2_loss_weighting:
                            accelerator.unscale_gradients()

                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                            torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        if args.edm2_loss_weighting:
                            # Use edm2-specific grad norm if provided, otherwise use max_grad_norm
                            edm2_grad_norm = (args.edm2_loss_weighting_max_grad_norm
                                             if args.edm2_loss_weighting_max_grad_norm is not None
                                             else args.max_grad_norm)
                            if edm2_grad_norm != 0.0:
                                edm2_params = list(accelerator.unwrap_model(edm2_model).parameters())
                                torch.nn.utils.clip_grad_norm_(edm2_params, edm2_grad_norm)

                        #if hasattr(network, "update_grad_norms"):
                        #    network.update_grad_norms()
                        #if hasattr(network, "update_norms"):
                        #    network.update_norms()

                    if _use_step_func:
                        # Schedulefree-plus optimizers (e.g. AdamWScheduleFreePlus) use
                        # step_func(function_value) instead of step(). step_func reads p.grad
                        # directly, so gradients must be unscaled for mixed precision.
                        # The Polyak step size requires the current loss value.
                        if accelerator.sync_gradients:
                            if not (args.max_grad_norm != 0.0 or args.edm2_loss_weighting):
                                # Gradients weren't unscaled above; do it now so step_func
                                # sees correctly-scaled gradients.
                                accelerator.unscale_gradients()
                            avg_loss = _step_func_loss_accum / accumulation_counter
                            _raw_optimizer.step_func(avg_loss)
                            _step_func_loss_accum = 0.0
                            # accelerate's optimizer.step() normally calls GradScaler.step()
                            # + update(), but step_func bypasses this. Update the scaler to
                            # reset its per-optimizer "unscaled" state so the next
                            # unscale_gradients() call doesn't raise RuntimeError, and to
                            # allow the scale factor to be adjusted.
                            scaler = getattr(optimizer, 'scaler', None)
                            if scaler is not None:
                                scaler.update()
                    else:
                        optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Weight noising: inject Gaussian noise into adapter params
                    # after optimizer step (inspired by ai-toolkit-perceptual).
                    # Runs after EMA would be updated so EMA stays clean.
                    if self.weight_noise_enabled and accelerator.sync_gradients:
                        unwrapped_network = accelerator.unwrap_model(network)
                        if hasattr(unwrapped_network, "inject_weight_noise"):
                            # Compute effective batch size and fallback LR for dynamic scaling.
                            # Use actual accumulated samples (handles incomplete batches
                            # and partial gradient accumulation) × num_processes for DDP.
                            _eff_bs = accumulated_samples * accelerator.num_processes
                            _fallback_lr = lr_scheduler.get_last_lr()[0]
                            _raw_opt = getattr(optimizer, 'optimizer', optimizer)
                            noise_norm = unwrapped_network.inject_weight_noise(
                                lr=_fallback_lr, effective_batch_size=_eff_bs,
                                optimizer=_raw_opt,
                            )
                            if current_global_step_wnoise is not None:
                                current_global_step_wnoise += noise_norm

                    if args.edm2_loss_weighting:
                        edm2_optimizer.step()
                        edm2_lr_scheduler.step()
                        # swap to pre_scaling_loss for logging
                        edm2_optimizer.zero_grad(set_to_none=True)

                    # Adaptive timestep sampling: run Algorithm 2 after optimizer step
                    if self.adaptive_manager is not None and accelerator.sync_gradients:
                        self.compute_adaptive_delta_after_step(unet, noise_scheduler, weight_dtype, accelerator, network)



                if args.scale_weight_norms and accelerator.sync_gradients:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    mean_grad_norm = None
                    mean_combined_norm = None
                    mean_norm = mean_norm.detach().item() if isinstance(mean_norm, torch.Tensor) else mean_norm
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    #if hasattr(network, "weight_norms"):
                    #    weight_norms = network.weight_norms()
                    #    mean_norm = weight_norms.mean().item() if weight_norms is not None else None
                    #    grad_norms = network.grad_norms()
                    #    mean_grad_norm = grad_norms.mean().item() if grad_norms is not None else None
                    #    combined_weight_norms = network.combined_weight_norms()
                    #    mean_combined_norm = combined_weight_norms.mean().item() if combined_weight_norms is not None else None
                    #    maximum_norm = weight_norms.max().item() if weight_norms is not None and weight_norms.numel() > 0 else None
                    #    keys_scaled = None
                    #    max_mean_logs = {}
                    # else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None
                    mean_grad_norm = None
                    mean_combined_norm = None
                    max_mean_logs = {}

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    rate_tracker.tick()
                    progress_bar.update(1)
                    global_step += 1

                    if (train_util.sample_images_check(args, None, global_step) or 
                        train_util.calculate_val_loss_check(args, global_step, step, val_dataloader, train_dataloader) or 
                        args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0):
                        with torch.no_grad():
                            accelerator.unwrap_model(network).eval()
                            if args.gradient_checkpointing:
                                accelerator.unwrap_model(unet).eval()
                                for t_enc in text_encoders:
                                    accelerator.unwrap_model(t_enc).eval()

                            optimizer_eval_fn()
                            self.sample_images(
                                accelerator, args, None, global_step, accelerator.device, vae, tokenizers, text_encoder, unet
                            )

                            if train_util.calculate_val_loss_check(args, global_step, step, val_dataloader, train_dataloader):
                                current_val_loss, average_val_loss, val_logs = self.calculate_val_loss(global_step, step, 
                                                                                                        skipped_dataloader or train_dataloader, 
                                                                                                        val_loss_recorder, 
                                                                                                        val_dataloader, 
                                                                                                        cyclic_val_dataloader, 
                                                                                                        network,
                                                                                                        tokenize_strategy, 
                                                                                                        text_encoders, 
                                                                                                        text_encoding_strategy, 
                                                                                                        unet, 
                                                                                                        vae, 
                                                                                                        noise_scheduler, 
                                                                                                        vae_dtype, 
                                                                                                        weight_dtype, 
                                                                                                        accelerator, 
                                                                                                        args, 
                                                                                                        current_epoch.value,
                                                                                                        batch,
                                                                                                        train_text_encoder)
                            else:
                                current_val_loss, average_val_loss, val_logs = None, None, {}
                            progress_bar.unpause()

                            # 指定ステップごとにモデルを保存
                            if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                                accelerator.wait_for_everyone()
                                if accelerator.is_main_process:
                                    ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                                    if args.edm2_loss_weighting:
                                        loss_weights_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step, "_edm2_loss_weights")
                                        save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step, epoch, dtype_override=torch.float32)

                                    if args.save_state:
                                        train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                                    remove_step_no = train_util.get_remove_step_no(args, global_step)
                                    if remove_step_no is not None:
                                        remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                        remove_model(remove_ckpt_name)

                                        if args.edm2_loss_weighting:
                                            remove_loss_weights_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no, "_edm2_loss_weights")
                                            remove_model(remove_loss_weights_ckpt_name)

                            optimizer_train_fn()
                            accelerator.unwrap_model(network).train()
                            if args.gradient_checkpointing:
                                accelerator.unwrap_model(unet).train()
                                for t_enc in text_encoders:
                                    accelerator.unwrap_model(t_enc).train()
                    else:
                        current_val_loss, average_val_loss, val_logs = None, None, None

                    # EDM2 graph generation - moved outside the sample/val/save conditional
                    if plot_edm2_loss_weighting_check(args, global_step):
                        plot_edm2_loss_weighting(args, global_step, edm2_model, noise_scheduler.config.num_train_timesteps, accelerator.device)

                current_global_step_loss += loss.detach().item()
                if args.edm2_loss_weighting:
                    current_global_step_loss_scaled += loss_scaled.detach().item()
                    current_global_step_loss_edm2 += edm2_loss.detach().item()
                else:
                    current_global_step_loss_scaled = None
                    current_global_step_loss_edm2 = None
                # Note: FFL loss is already tracked above (accumulated per micro-step)

                if accelerator.sync_gradients:
                    loss_recorder.add(current_global_step_loss / accumulation_counter)
                    if args.edm2_loss_weighting:
                        loss_scaled_recorder.add(current_global_step_loss_scaled / accumulation_counter)
                        loss_edm2_recorder.add(current_global_step_loss_edm2 / accumulation_counter)
                        
                    avr_loss: float = loss_recorder.average
                    combined = {**max_mean_logs, "avr_loss": f"{avr_loss:.4f}"}
                    progress_bar.set_postfix_str(f"{rate_tracker.display_rate}, " + ", ".join(f"{k}={v}" for k, v in combined.items()))

                    if is_tracking:
                        current_global_step_loss = (current_global_step_loss / accumulation_counter)
                        if args.edm2_loss_weighting:
                            current_global_step_loss_scaled = (current_global_step_loss_scaled / accumulation_counter)
                            average_loss_scaled: float = loss_scaled_recorder.average
                            current_global_step_loss_edm2 = (current_global_step_loss_edm2 / accumulation_counter)
                            average_loss_edm2: float = loss_edm2_recorder.average
                        else:
                            current_global_step_loss_scaled = None
                            average_loss_scaled = None
                            current_global_step_loss_edm2 = None
                            average_loss_edm2 = None

                        logs = self.generate_step_logs(
                            args,
                            current_global_step_loss,
                            avr_loss,
                            lr_scheduler,
                            lr_descriptions,
                            optimizer,
                            keys_scaled,
                            mean_norm,
                            maximum_norm,
                            mean_grad_norm,
                            mean_combined_norm,
                            edm2_lr_scheduler,
                            current_global_step_loss_scaled,
                            average_loss_scaled,
                            current_global_step_loss_edm2,
                            average_loss_edm2,
                            current_val_loss=current_val_loss,
                            average_val_loss=average_val_loss,
                            current_ffl_loss=(current_global_step_ffl / accumulation_counter) if self.ffl_enabled and current_global_step_ffl is not None else None,
                            current_patch_topology_loss=(current_global_step_patch_topo / accumulation_counter) if self.patch_topology_enabled and current_global_step_patch_topo is not None else None,
                            current_patch_topology_weight=self.patch_topology_effective_weight if self.patch_topology_enabled else None,
                            current_wav_mask_ratio=(current_global_step_wav_mask / accumulation_counter) if self.wavelet_masking_enabled and current_global_step_wav_mask is not None else None,
                            current_weight_noise_norm=(current_global_step_wnoise / accumulation_counter) if self.weight_noise_enabled and current_global_step_wnoise is not None else None,
                            current_hf_loss=(current_global_step_hf / accumulation_counter) if self.hf_scale > 0.0 and current_global_step_hf is not None else None,
                            current_anchor_loss=(current_global_step_anchor / accumulation_counter) if self.anchor_scale > 0.0 and current_global_step_anchor is not None else None,
                            current_adv_loss=(current_global_step_adv / accumulation_counter) if self.adv_scale > 0.0 and current_global_step_adv is not None else None,
                            it_s=rate_tracker.it_per_sec,
                        )
                        if val_logs:
                            logs.update(val_logs)
                        accelerator.log(logs, step=global_step)
                    current_global_step_loss = 0.0
                    if args.edm2_loss_weighting:
                        current_global_step_loss_scaled = 0.0
                        current_global_step_loss_edm2 = 0.0
                    if self.ffl_enabled:
                        current_global_step_ffl = 0.0
                    if self.patch_topology_enabled:
                        current_global_step_patch_topo = 0.0
                    if self.wavelet_masking_enabled:
                        current_global_step_wav_mask = 0.0
                    if self.weight_noise_enabled:
                        current_global_step_wnoise = 0.0
                    if self.hf_scale > 0.0:
                        current_global_step_hf = 0.0
                    if self.anchor_scale > 0.0:
                        current_global_step_anchor = 0.0
                    if self.adv_scale > 0.0:
                        current_global_step_adv = 0.0
                    accumulation_counter = 0
                    accumulated_samples = 0

                if global_step >= args.max_train_steps:
                    break

            # END OF EPOCH
            if is_tracking:
                logs = {"loss/epoch_average": loss_recorder.average}
                accelerator.log(logs, step=global_step)

            accelerator.wait_for_everyone()

            if (train_util.sample_images_check(args, current_epoch.value, global_step) or 
                args.save_every_n_epochs is not None):
                with torch.no_grad():
                    # 指定エポックごとにモデルを保存
                    optimizer_eval_fn()
                    accelerator.unwrap_model(network).eval()
                    if args.gradient_checkpointing:
                        accelerator.unwrap_model(unet).eval()
                        for t_enc in text_encoders:
                            accelerator.unwrap_model(t_enc).eval()

                    if args.save_every_n_epochs is not None:
                        saving = (current_epoch.value) % args.save_every_n_epochs == 0 and (current_epoch.value) < num_train_epochs
                        if is_main_process and saving:
                            ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, current_epoch.value)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, current_epoch.value)

                            if args.edm2_loss_weighting:
                                loss_weights_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, current_epoch.value, "_edm2_loss_weights")
                                save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step, current_epoch.value, dtype_override=torch.float32)

                            remove_epoch_no = train_util.get_remove_epoch_no(args, current_epoch.value)
                            if remove_epoch_no is not None:
                                remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                                remove_model(remove_ckpt_name)

                                if args.edm2_loss_weighting:
                                    remove_loss_weights_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no, "_edm2_loss_weights")
                                    remove_model(remove_loss_weights_ckpt_name)

                            if args.save_state:
                                train_util.save_and_remove_state_on_epoch_end(args, accelerator, current_epoch.value)

                    self.sample_images(accelerator, args, current_epoch.value, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
                    progress_bar.unpause()
                    optimizer_train_fn()
                    accelerator.unwrap_model(network).train()
                    if args.gradient_checkpointing:
                        accelerator.unwrap_model(unet).train()
                        for t_enc in text_encoders:
                            accelerator.unwrap_model(t_enc).train()

            # end of epoch

            # Release CUDA caching-allocator reserved memory accumulated during
            # this epoch (distinct bucket-shape pools from random-crop
            # training, validation, and sample-image generation).
            clean_memory_on_device(accelerator.device)

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()
        optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

            if args.edm2_loss_weighting:
                loss_weights_ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as, "_edm2_loss_weights")
                save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step, num_train_epochs, force_sync_upload=True, dtype_override=torch.float32)

            logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing for U-Net or DiT, if supported"
        " / 勾配チェックポイント時にテンソルをCPUにオフロードする（U-NetまたはDiTのみ、サポートされている場合）",
    )
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )
    parser.add_argument(
        "--disable_cross_attn_mask",
        action="store_true",
        help="Disable SDXL cross-attention masking so padded tokens participate normally / SDXLのcross-attentionマスク機能を無効化する",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        nargs="*",
        help="learning rate for Text Encoder, can be multiple / Text Encoderの学習率、複数指定可能",
    )
    parser.add_argument(
        "--fp8_base_unet",
        action="store_true",
        help="use fp8 for U-Net (or DiT), Text Encoder is fp16 or bf16"
        " / U-Net（またはDiT）にfp8を使用する。Text Encoderはfp16またはbf16",
    )

    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached / initial_stepに到達するまで学習をスキップする",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + " / 初期エポック数、1で最初のエポック（未指定時と同じ）。注意：initial_epoch/stepはlr schedulerに影響しないため、`--resume`しない場合はlr schedulerは0から始まる",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + " / 初期ステップ数、全エポックを含むステップ数、0で最初のステップ（未指定時と同じ）。initial_epochを上書きする",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help="Validation seed for shuffling validation dataset, training `--seed` used otherwise / 検証データセットをシャッフルするための検証シード、それ以外の場合はトレーニング `--seed` を使用する",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for validation images out of the training dataset / 学習画像から検証画像に分割する割合",
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=None,
        help="Run validation on validation dataset every N steps. By default, validation will only occur every epoch if a validation dataset is available / 検証データセットの検証をNステップごとに実行します。デフォルトでは、検証データセットが利用可能な場合にのみ、検証はエポックごとに実行されます",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=None,
        help="Run validation dataset every N epochs. By default, validation will run every epoch if a validation dataset is available / 検証データセットをNエポックごとに実行します。デフォルトでは、検証データセットが利用可能な場合、検証はエポックごとに実行されます",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Max number of validation dataset items processed. By default, validation will run the entire validation dataset / 処理される検証データセット項目の最大数。デフォルトでは、検証は検証データセット全体を実行します",
    )

    parser.add_argument(
        "--validation_timesteps",
        type=str,
        default=r"[50, 350, 500, 650, 950]",
        help="A list of timesteps to use for each validation step."
    )  

    parser.add_argument(
        "--use_ramtorch_network",
        action="store_true",
        help="Use RamTorch to reduce GPU memory usage by keeping network/lora linear weights in system RAM. " \
        "Requires use of optimizers that have been modified to support it, currently only SimplifiedAdEMAMix, SimplifiedAdEMAMixExM, and OCGOpt.",
    )

    parser.add_argument(
        "--edm2_loss_weighting",
        action="store_true",
        help="Use EDM2 loss weighting.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_laplace",
        action="store_true",
        help="Use EDM2 loss weighting to calculate timestep sampling using laplace.",
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
        default=r"{'weight_decay': 0, 'betas': (0.9,0.999)}",
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
        default=0.1,
        help="Percent of training steps to use for warmup.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_lr_scheduler_constant_percent",
        type=float,
        default=0.1,
        help="Percent of training steps to maintain constant LR before decay.",
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
        6 is a good value for v-pred + ztsnr without any augmentation (i.e. low min snr gamma, debiased loss, or scaled v-pred loss). 
        If any of the noted augmentations are used, weighting values can reach ~100-150.""",
    )

    parser.add_argument(
        "--edm2_loss_weighting_generate_graph_y_scale",
        type=str,
        default="linear",
        choices=["linear", "log"],
        help="""Select between linear or log scaling for the y-axis.""",
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
        "--edm2_loss_weighting_importance_weighting",
        action="store_true",
        help="If edm2 loss scaling weights are weighted by importance, which is based using a specific min snr gamma value and SNR for the given timestep. " \
        "Default behavior when edm2_loss_weighting_importance_weighting is enabled is to disable normal min snr gamma and debiased loss if enabled." \
        "It is not advised to stack with either, as there is a possiblity of loss curving to 0 as SNR approaches 0." \
        "If you still wish to, set edm2_loss_weighting_importance_weighting_safety_override=True at your own risk."
    )

    parser.add_argument(
        "--edm2_loss_weighting_importance_weighting_max",
        type=float,
        default=10.0,
        help="The max loss weighting/scaling to apply when using edm2 importance weighting, has no effect otherwise.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_importance_min_snr_gamma",
        type=float,
        default=1.0,
        help="The min snr gamma used for edm2 importance weighting as a heuristic, has no effect if not using importance weighting. " \
        "Not related to the typical application of min snr gamma.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_importance_weighting_safety_override",
        action="store_true",
        help="At your own risk, you may set this to true to ALLOW stacking debiased loss and/or typical min snr gamma with EDM2 using importance weighting.",
    )

    parser.add_argument(
        "--edm2_loss_weighting_max_grad_norm",
        type=float,
        default=None,
        help="Maximum gradient norm for EDM2 loss weighting model. If not specified, uses --max_grad_norm value. Set to 0 to disable clipping for EDM2. / EDM2損失重み付けモデルの最大勾配ノルム。指定しない場合は--max_grad_normの値を使用。0に設定するとEDM2のクリッピングを無効化。"
    )

    parser.add_argument(
        "--orthograd_targets",
        type=str,
        default=r"['lora_down.weight','lora_up.weight','lora_down1.weight','lora_up1.weight','lora_down2.weight','lora_up2.weight','a1.weight','a2.weight','b1.weight','b2.weight','c1.weight']",
        help="A list of strings to determine which named parameters should subject to orthgrad, based on their name containing the string."
    )

    parser.add_argument(
        "--pin_data_loader_memory",
        action="store_true",
        help="Pins dataloader memory, may speed up dataloader operations.",
    )

    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for faster GPU loading / GPU の読み込みを高速化するためのピンメモリ",
    )

    parser.add_argument(
        "--differential_guidance",
        action="store_true",
        help="Differential Guidance amplifies applies an amplification between the difference of the model prediction and the target during training to make " \
        "a new target. This may help improve convergence to the actual target. " \
        "See original code at https://github.com/ostris/ai-toolkit/commit/2e7b2d9926de40a7b9119322c1d8fc085b1283e4#diff-fb148217f864741f0e90717dc8ab38dff83a42e917a20540f65afb1c3aedaa85",
    )

    parser.add_argument(
        "--differential_guidance_scale",
        type=float,
        default=3.0,
        help="Differential Guidance Scale is used to determine the multiplier of the difference of the model prediction and the target. " \
        "--differential_guidance arg must be passed for this to be applied. " \
        "See original code at https://github.com/ostris/ai-toolkit/commit/2e7b2d9926de40a7b9119322c1d8fc085b1283e4#diff-fb148217f864741f0e90717dc8ab38dff83a42e917a20540f65afb1c3aedaa85",
    )


    parser.add_argument(
        "--vae_reflection_padding",
        action="store_true",
        help="switch VAE convolutions to reflection padding (improves border quality for some custom VAEs) / VAEの畳み込みを反射パディングに切り替える",
    )
    parser.add_argument(
        "--vae_custom_scale",
        type=float,
        default=None,
        help="override the latent scaling factor applied after VAE encode / VAEエンコード後のスケーリング係数を上書きする",
    )
    parser.add_argument(
        "--vae_custom_shift",
        type=float,
        default=None,
        help="apply a constant latent shift before scaling (e.g. Flux-style offset) / スケーリング前に潜在表現へ定数シフトを適用する",
    )

    parser.add_argument(
        "--flow_model",
        action="store_true",
        help="enable Rectified Flow training objective instead of standard diffusion / 通常の拡散ではなくRectified Flowで学習する",
    )
    parser.add_argument(
        "--flow_use_ot",
        action="store_true",
        help="pair latents and noise with cosine optimal transport when using Rectified Flow / Rectified Flow使用時にOTでlatentとノイズを対応付ける",
    )
    parser.add_argument(
        "--flow_timestep_distribution",
        type=str,
        default="logit_normal",
        choices=["logit_normal", "uniform"],
        help="sampling distribution over Rectified Flow sigmas (default: logit_normal) / Rectified Flowのシグマの分布（デフォルトlogit_normal）",
    )
    parser.add_argument(
        "--flow_logit_mean",
        type=float,
        default=0.0,
        help="mean of the logit-normal distribution when using Rectified Flow / Rectified Flowでlogit-normal分布を用いるときの平均値",
    )
    parser.add_argument(
        "--flow_logit_std",
        type=float,
        default=1.0,
        help="stddev of the logit-normal distribution when using Rectified Flow / Rectified Flowでlogit-normal分布を用いるときの標準偏差",
    )
    parser.add_argument(
        "--flow_uniform_shift",
        action="store_true",
        help="apply resolution-dependent shift to Rectified Flow timesteps (SD3-style) / Rectified Flowタイムステップに解像度依存のシフトを適用する",
    )
    parser.add_argument(
        "--flow_uniform_base_pixels",
        type=float,
        default=1024.0 * 1024.0,
        help="reference pixel count used for the resolution-dependent timestep shift / タイムステップシフトで使用する基準ピクセル数",
    )
    parser.add_argument(
        "--flow_uniform_static_ratio",
        type=float,
        default=None,
        help="use a fixed sqrt(m/n) ratio (e.g. 2.5) for Rectified Flow timestep shift; overrides resolution-based shift / 一定のsqrt(m/n)比率（例:2.5）でRectified Flowタイムステップをシフトする（解像度依存シフトを上書き）",
    )
    parser.add_argument(
        "--contrastive_flow_matching",
        action="store_true",
        help="Enable Contrastive Flow Matching (ΔFM) objective. Works with v-parameterization or Rectified Flow.",
    )
    parser.add_argument(
        "--cfm_lambda",
        type=float,
        default=0.05,
        help="Lambda weight for the contrastive term in ΔFM loss (default: 0.05).",
    )
    parser.add_argument(
        "--use_zero_cond_dropout",
        type=bool,
        default=False,
        help="For full caption dropout, use zero conditioning instead of empty caption"
    )
    # parser.add_argument("--loraplus_lr_ratio", default=None, type=float, help="LoRA+ learning rate ratio")
    # parser.add_argument("--loraplus_unet_lr_ratio", default=None, type=float, help="LoRA+ UNet learning rate ratio")
    # parser.add_argument("--loraplus_text_encoder_lr_ratio", default=None, type=float, help="LoRA+ text encoder learning rate ratio")

    # Latent Wavelet Diffusion (LWD) arguments
    parser.add_argument(
        "--wavelet_masking",
        action="store_true",
        help="Enable LWD wavelet-based spatial masking on the training loss. "
        "Focuses training on detail-rich regions via frequency-aware masks derived from Haar DWT. "
        "Based on: 'Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis' (ICLR 2026). "
        "Requires pytorch-wavelets.",
    )
    parser.add_argument(
        "--wavelet_mask_l_bound",
        type=float,
        default=0.3,
        help="Lower bound l for wavelet masking (default: 0.3). All spatial regions receive at least "
        "l*T supervision steps. Paper ablation shows 0.3 is optimal. Range: [0.0, 1.0].",
    )

    # High-Frequency Token latent loss arguments (see library/hf_token_loss.py)
    parser.add_argument(
        "--hf_scale",
        type=float,
        default=0.0,
        help="High-Frequency token latent loss weight (lambda, 0 = off, must be >= 0). "
        "L_total = L_mse + hf_scale * L_hf. Concentrates training effort on image tokens "
        "carrying fine (high-frequency) detail.",
    )
    parser.add_argument(
        "--hf_exponent",
        type=float,
        default=1.0,
        help="HF token weight concentration exponent (gamma, must be > 0). 1 = linear in detail, "
        "> 1 concentrates on the highest-detail tokens, < 1 flattens toward uniform.",
    )
    parser.add_argument(
        "--hf_patch",
        type=int,
        default=2,
        help="HF token patch size; MUST equal the model's own patchify size. "
        "2 for latent-space models (SD1.5/SD2/SDXL/Flux/SD3/Lumina/Hunyuan/Anima), "
        "16 for pixel-space models (e.g. ChromaRadiance).",
    )
    parser.add_argument(
        "--hf_high_noise_snr_cut",
        type=float,
        default=1.0 / 9.0,
        help="Universal SNR cutoff for HF high-noise attenuation (default: 0.111111, "
        "equivalent to flow sigma 0.75). Must be > 0.",
    )
    parser.add_argument(
        "--hf_high_noise_min_weight",
        type=float,
        default=0.10,
        help="Minimum HF multiplier at the highest flow noise levels (default: 0.10). Range: [0, 1].",
    )
    parser.add_argument(
        "--hf_high_noise_power",
        type=float,
        default=1.0,
        help="Power controlling high-noise HF attenuation below hf_high_noise_snr_cut (default: 1.0).",
    )

    # Multiscale MSE x0-prediction ("anchor") loss arguments (see library/anchor_loss.py)
    parser.add_argument(
        "--anchor_scale",
        type=float,
        default=0.0,
        help="Multiscale MSE x0-prediction (anchor) loss weight (0 = off, must be >= 0). "
        "L_total = L_mse + anchor_scale * L_anchor. Frequency-even Laplacian-pyramid "
        "reconstruction term on the predicted clean output (x0): emphasizes composition "
        "and global look without zeroing out fine detail. Sweep 0.05 - 1.0.",
    )
    parser.add_argument(
        "--anchor_levels",
        type=int,
        default=4,
        help="Number of Laplacian pyramid band-pass octaves for the anchor loss (plus the "
        "coarsest Gaussian residual). Must be >= 1; clamped down to what the latent grid "
        "supports. More levels give subject/identity detail a voice, fewer levels judge "
        "mostly composition/global look.",
    )
    parser.add_argument(
        "--anchor_snr_weighting",
        action="store_true",
        help="Apply the soft SNR(t) gate to the anchor loss (batch-mean-1 normalized): "
        "early (noisy, low-SNR) steps weigh less, late (clean, high-SNR) steps weigh more. "
        "Inert while --anchor_scale is 0.",
    )

    # Stationary multi-scale RFF-MMD distribution-matching loss.
    parser.add_argument(
        "--adv_scale",
        type=float,
        default=0.0,
        help="Stationary multi-scale RFF-MMD loss weight (0 = off). Adds a distribution-matching "
        "term between each predicted-clean sample and a precomputed fixed training-set reference.",
    )
    parser.add_argument(
        "--adv_levels",
        type=int,
        default=4,
        help="Laplacian-pyramid octave levels for the stationary adv loss (plus coarsest residual).",
    )
    parser.add_argument(
        "--adv_domain",
        choices=("latent", "tae_pixels"),
        default="latent",
        help="Reference/prediction domain for adv matching. This train_network implementation currently supports latent domain.",
    )
    parser.add_argument(
        "--adv_features",
        type=int,
        default=256,
        help="RFF feature count per bandwidth for the stationary adv loss.",
    )
    parser.add_argument(
        "--adv_bandwidths",
        type=str,
        default="",
        help="Comma-separated positive RBF bandwidth ladder; empty uses the reference-token median heuristic (0.5x, 1x, 2x).",
    )
    parser.add_argument(
        "--adv_tokens_per_band",
        type=int,
        default=1024,
        help="Maximum adaptive-average-pooled spatial tokens per Laplacian band.",
    )
    parser.add_argument(
        "--adv_reference_samples",
        type=int,
        default=4096,
        help="Bounded per-band reference-token sample count for the bandwidth heuristic.",
    )
    parser.add_argument(
        "--adv_rff_seed",
        type=lambda value: int(value, 0),
        default=0x5A17C0DE,
        help="Dedicated seed for deterministic RFF frequencies/phases (accepts decimal or 0x-prefixed integer).",
    )
    parser.add_argument(
        "--adv_snr_weighting",
        action="store_true",
        help="Apply a detached, batch-mean-one SNR gate to the adv term; off by default.",
    )
    parser.add_argument(
        "--adv_cache_dir",
        type=str,
        default=None,
        help="Directory for the stationary adv reference cache (default: <output_dir>/adv_cache).",
    )

    parser.add_argument(
        "--keep_unet_dtype",
        action="store_true",
        help="TBD",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)
