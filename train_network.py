import importlib
import argparse
import math
import os
import sys
import random
import time
import json
import contextlib
from multiprocessing import Value
from typing import Any, List
import toml
from tools.grokfast import Gradfilter_ma, Gradfilter_ema
import numpy as np
import tools.edm2_loss_mm as edm2_loss_mm
import ast

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from accelerate import Accelerator, AutocastKwargs
from diffusers import DDPMScheduler
from library import deepspeed_utils, model_util, strategy_base, strategy_sd

import library.train_util as train_util
from library.train_util import DreamBoothDataset
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging
import itertools

import tools.stochastic_accumulator as stochastic_accumulator

logger = logging.getLogger(__name__)

@torch.no_grad()
def analyze_gradient_norms(parameters):
    """
    Comprehensive analysis of gradient norms from model parameters.
    
    Args:
        parameters: An iterable of parameters with gradients (e.g., model.parameters())
        
    Returns:
        dict: Dictionary containing various gradient statistics
    """
    # Extract gradient norms and zero gradients count
    grad_norms = []
    
    for param in parameters:
        if param.grad is not None:
            grad = param.grad

            norm = torch.norm(grad, p=2.0).item()
            if not (np.isnan(norm) or np.isinf(norm)):
                grad_norms.append(norm)
    
    if not grad_norms:
        return {
                'train/grad_norm/mean': 0.0,
                'train/grad_norm/median': 0.0,
                'train/grad_norm/std': 0.0,
                'train/grad_norm/min': 0.0,
                'train/grad_norm/max': 0.0,
                'train/grad_norm/p10': 0.0,
                'train/grad_norm/p25': 0.0,
                'train/grad_norm/p75': 0.0,
                'train/grad_norm/p90': 0.0,
                'train/grad_norm/p95': 0.0,
                'train/grad_norm/p98': 0.0,
                'train/grad_norm/p99': 0.0,
                'train/grad_norm/p995': 0.0,
                'train/grad_norm/p998': 0.0,
                'train/grad_norm/p999': 0.0,
            }
    
    # Convert to numpy array for calculations
    grad_norms = np.array(grad_norms)
    
    # Basic statistics
    stats = {
        'train/grad_norm/mean': float(np.mean(grad_norms)),
        'train/grad_norm/median': float(np.median(grad_norms)),
        'train/grad_norm/std': float(np.std(grad_norms)),
        'train/grad_norm/min': float(np.min(grad_norms)),
        'train/grad_norm/max': float(np.max(grad_norms)),
        'train/grad_norm/p10': float(np.percentile(grad_norms, 10)),  # Lower tail
        'train/grad_norm/p25': float(np.percentile(grad_norms, 25)),  # First quartile
        'train/grad_norm/p75': float(np.percentile(grad_norms, 75)),  # Third quartile
        'train/grad_norm/p90': float(np.percentile(grad_norms, 90)),
        'train/grad_norm/p95': float(np.percentile(grad_norms, 95)),
        'train/grad_norm/p98': float(np.percentile(grad_norms, 98)),
        'train/grad_norm/p99': float(np.percentile(grad_norms, 99)),
        'train/grad_norm/p995': float(np.percentile(grad_norms, 99.5)),
        'train/grad_norm/p998': float(np.percentile(grad_norms, 99.8)),
        'train/grad_norm/p999': float(np.percentile(grad_norms, 99.9)),
    }
    
    return stats

@torch.no_grad()
def analyze_model_norms(unscaled_norms, scaled_norms):

    if not scaled_norms or not unscaled_norms:
        return {
                'model/module_norm/mean': 0.0,
                'model/module_norm/median': 0.0,
                'model/module_norm/std': 0.0,
                'model/module_norm/min': 0.0,
                'model/module_norm/max': 0.0,
                'model/module_norm/p10': 0.0,
                'model/module_norm/p25': 0.0,
                'model/module_norm/p75': 0.0,
                'model/module_norm/p90': 0.0,
                'model/module_norm/p95': 0.0,
                'model/module_norm/p98': 0.0,
                'model/module_norm/p99': 0.0,
                'model/module_norm/p995': 0.0,
                'model/module_norm/p998': 0.0,
                'model/module_norm/p999': 0.0,
                'model/module_norm/unscaled/mean': 0.0,
                'model/module_norm/unscaled/median': 0.0,
                'model/module_norm/unscaled/std': 0.0,
                'model/module_norm/unscaled/min': 0.0,
                'model/module_norm/unscaled/max': 0.0,
                'model/module_norm/unscaled/p10': 0.0,
                'model/module_norm/unscaled/p25': 0.0,
                'model/module_norm/unscaled/p75': 0.0,
                'model/module_norm/unscaled/p90': 0.0,
                'model/module_norm/unscaled/p95': 0.0,
                'model/module_norm/unscaled/p98': 0.0,
                'model/module_norm/unscaled/p99': 0.0,
                'model/module_norm/unscaled/p995': 0.0,
                'model/module_norm/unscaled/p998': 0.0,
                'model/module_norm/unscaled/p999': 0.0,
            }
    
    # Convert to numpy array for calculations
    unscaled_norms = np.array(unscaled_norms)
    scaled_norms = np.array(scaled_norms)
    
    # Basic statistics
    stats = {
        'model/module_norm/mean': float(np.mean(scaled_norms)),
        'model/module_norm/median': float(np.median(scaled_norms)),
        'model/module_norm/std': float(np.std(scaled_norms)),
        'model/module_norm/min': float(np.min(scaled_norms)),
        'model/module_norm/max': float(np.max(scaled_norms)),
        'model/module_norm/p10': float(np.percentile(scaled_norms, 10)),  # Lower tail
        'model/module_norm/p25': float(np.percentile(scaled_norms, 25)),  # First quartile
        'model/module_norm/p75': float(np.percentile(scaled_norms, 75)),  # Third quartile
        'model/module_norm/p90': float(np.percentile(scaled_norms, 90)),
        'model/module_norm/p95': float(np.percentile(scaled_norms, 95)),
        'model/module_norm/p98': float(np.percentile(scaled_norms, 98)),
        'model/module_norm/p99': float(np.percentile(scaled_norms, 99)),
        'model/module_norm/p995': float(np.percentile(scaled_norms, 99.5)),
        'model/module_norm/p998': float(np.percentile(scaled_norms, 99.8)),
        'model/module_norm/p999': float(np.percentile(scaled_norms, 99.9)),
        'model/module_norm/unscaled/mean': float(np.mean(unscaled_norms)),
        'model/module_norm/unscaled/median': float(np.median(unscaled_norms)),
        'model/module_norm/unscaled/std': float(np.std(unscaled_norms)),
        'model/module_norm/unscaled/min': float(np.min(unscaled_norms)),
        'model/module_norm/unscaled/max': float(np.max(unscaled_norms)),
        'model/module_norm/unscaled/p10': float(np.percentile(unscaled_norms, 10)),  # Lower tail
        'model/module_norm/unscaled/p25': float(np.percentile(unscaled_norms, 25)),  # First quartile
        'model/module_norm/unscaled/p75': float(np.percentile(unscaled_norms, 75)),  # Third quartile
        'model/module_norm/unscaled/p90': float(np.percentile(unscaled_norms, 90)),
        'model/module_norm/unscaled/p95': float(np.percentile(unscaled_norms, 95)),
        'model/module_norm/unscaled/p98': float(np.percentile(unscaled_norms, 98)),
        'model/module_norm/unscaled/p99': float(np.percentile(unscaled_norms, 99)),
        'model/module_norm/unscaled/p995': float(np.percentile(unscaled_norms, 99.5)),
        'model/module_norm/unscaled/p998': float(np.percentile(unscaled_norms, 99.8)),
        'model/module_norm/unscaled/p999': float(np.percentile(unscaled_norms, 99.9)),
    }
    
    return stats

class NetworkTrainer:
    def __init__(self):
        self.vae_scale_factor: float = 0.18215
        self.is_sdxl: bool = False

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
        grad_norm=None,
        grad_norm_clipped=None,
        current_val_loss=None,
        average_val_loss=None,
        current_loss_scaled=None,
        average_loss_scaled=None,
        edm2_grad_norm=None,
        edm2_grad_norm_clipped=None,
        edm2_lr_scheduler=None,
        gradient_stats=None,
        network_norm_stats=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if current_loss_scaled is not None:
            logs["loss/current_scaled"] = current_loss_scaled
            logs["loss/average_scaled"] = average_loss_scaled

        if grad_norm is not None:
            logs["train/grad_norm"] = grad_norm
            logs["train/grad_norm_clipped"] = grad_norm_clipped

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        if current_val_loss is not None:
            logs["loss/current_val_loss"] = current_val_loss                      
            logs["loss/average_val_loss"] = average_val_loss

        if edm2_grad_norm is not None:
            logs["train/edm2_grad_norm"] = edm2_grad_norm
            logs["train/edm2_grad_norm_clipped"] = edm2_grad_norm_clipped

        if gradient_stats:
            logs = {**logs, **gradient_stats}

        if network_norm_stats:
            logs = {**logs, **network_norm_stats}

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else -1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{idx}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                # tracking d*lr value
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )
            if (
                (args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) or args.optimizer_type.lower().endswith("ProdigyPlusExMachinaScheduleFree".lower())) and optimizer is not None
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = (
                    optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                )
        else:
            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )
                if (
                    (args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) or args.optimizer_type.lower().endswith("ProdigyPlusExMachinaScheduleFree".lower())) and optimizer is not None
                ):  
                    logs[f"lr/d*lr/group{i}"] = (
                        optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]
                    )

        if edm2_lr_scheduler is not None:
            logs[f"lr/edm2"] = edm2_lr_scheduler.get_last_lr()[0]


        return logs

    def assert_extra_args(self, args, train_dataset_group):
        train_dataset_group.verify_bucket_reso_steps(64)

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def get_tokenize_strategy(self, args):
        return strategy_sd.SdTokenizeStrategy(args.v2, args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd.SdTokenizeStrategy) -> List[Any]:
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            True, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
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

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        for t_enc in text_encoders:
            t_enc.to(device=accelerator.device, dtype=weight_dtype)

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype, **kwargs):
        noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def all_reduce_network(self, accelerator, network):
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

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

        # Pass mu and b if user provided them
        prepare_scheduler_for_custom_training(
            noise_scheduler,
            device,
            mu=args.laplace_timestep_sampling_mu,
            b=args.laplace_timestep_sampling_b
        )

        return noise_scheduler

    def encode_images_to_latents(self, args, accelerator, vae, images):
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, args, latents):
        return latents * self.vae_scale_factor
    
    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        fixed_timesteps=None,
        train=True
    ):
        if args.loss_related_use_float64:
            # Convert to float64, noise and noisy latents will be float64 due to using like on latents
            latents = latents.to(torch.float64)

        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, fixed_timesteps, train)

        # ensure the hidden state will require grad
        if train and args.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)

        # Predict the noise residual
        with accelerator.autocast():
            noise_pred = self.call_unet(
                args,
                accelerator,
                unet,
                noisy_latents.requires_grad_(train and train_unet),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
            )

        if args.loss_related_use_float64:
            noise_pred = noise_pred.to(torch.float64)

        if args.v_parameterization:
            # v-parameterization training
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        if args.loss_related_use_float64:
            target = target.to(torch.float64)

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
                        batch,
                        weight_dtype,
                        indices=diff_output_pr_indices,
                    )
                network.set_multiplier(1.0)  # may be overwritten by "network_multipliers" in the next step
                target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

        return noise_pred, target, timesteps, None, noisy_latents
    
    def determine_grad_sync_context(self, accelerator, sync_gradients, training_model, lossweightMLP = None):
        if not sync_gradients and accelerator.num_processes > 1:
            if lossweightMLP is not None:
                return accelerator.no_sync(training_model, lossweightMLP)
            else:
                return accelerator.no_sync(training_model)
        else:
            return contextlib.nullcontext()
        
    def post_process_loss(self, loss, args, timesteps, noise_scheduler, train=True):
        if args.min_snr_gamma and train and not args.sangoi_loss_modifier:
            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
        if args.scale_v_pred_loss_like_noise_pred and train:
            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
        if args.v_pred_like_loss and train:
            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
        if args.debiased_estimation_loss and train:
            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)
        return loss

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

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        pass

    # endregion

    def plot_dynamic_loss_weighting_check(self, args, global_step):
        return args.edm2_loss_weighting and args.edm2_loss_weighting_generate_graph and (global_step % (int(args.edm2_loss_weighting_generate_graph_every_x_steps) if args.edm2_loss_weighting_generate_graph_every_x_steps else 20) == 0 or global_step >= args.max_train_steps)

    def process_val_batch(self, network, batch, tokenizers, tokenize_strategy, text_encoders, 
                          text_encoding_strategy, unet, vae, noise_scheduler, vae_dtype, weight_dtype, 
                          accelerator, args, timesteps_list: list = [10, 350, 500, 650, 990], train_text_encoder: bool = True):
        total_loss = 0.0 
        with torch.autograd.grad_mode.inference_mode(mode=True):
            if "latents" in batch and batch["latents"] is not None:
                latents = batch["latents"].to(device=accelerator.device, dtype=weight_dtype)
            else:
                if args.cache_latents:
                    clean_memory_on_device(accelerator.device)
                    vae.to(device=accelerator.device, dtype=vae_dtype)
                    vae.requires_grad_(False)
                    vae.eval()

                # latentに変換
                latents = self.encode_images_to_latents(args, accelerator, vae, batch["images"].to(device=vae.device, dtype=vae_dtype))
                latents = latents.to(dtype=weight_dtype)

                # NaNが含まれていれば警告を表示し0に置き換える
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = torch.nan_to_num(latents, 0, out=latents)

                batch["latents"] = latents
                latents = latents.to(device=accelerator.device)

                if args.cache_latents:
                    vae.to(device="cpu")
                    clean_memory_on_device(accelerator.device)

            latents = self.shift_scale_latents(args, latents)

            network_has_multiplier = hasattr(network, "set_multiplier")
            # get multiplier for each sample
            if network_has_multiplier:
                multipliers = batch["network_multipliers"]
                # if all multipliers are same, use single multiplier
                if torch.all(multipliers == multipliers[0]):
                    multipliers = multipliers[0].item()
                else:
                    raise NotImplementedError("multipliers for each sample is not supported yet")
                # print(f"set multiplier: {multipliers}")
                accelerator.unwrap_model(network).set_multiplier(multipliers)

            text_encoder_conds = []
            text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
            if text_encoder_outputs_list is not None:
                text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs
            if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
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
                        input_ids = [ids.to(device=accelerator.device) for ids in batch["input_ids_list"]]
                        encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(args, accelerator, text_encoders),
                            input_ids,
                        )
                        if args.full_fp16:
                            encoded_text_encoder_conds = [c.to(dtype=weight_dtype) for c in encoded_text_encoder_conds]

                # if text_encoder_conds is not cached, use encoded_text_encoder_conds
                if len(text_encoder_conds) == 0:
                    text_encoder_conds = encoded_text_encoder_conds
                else:
                    # if encoded_text_encoder_conds is not None, update cached text_encoder_conds
                    for i in range(len(encoded_text_encoder_conds)):
                        if encoded_text_encoder_conds[i] is not None:
                            text_encoder_conds[i] = encoded_text_encoder_conds[i]

            batch_size = latents.shape[0]
            for fixed_timesteps in timesteps_list:
                with accelerator.autocast():
                    timesteps = torch.full((batch_size,), fixed_timesteps, dtype=torch.long, device=latents.device)

                    # Get noise prediction and target
                    noise_pred, target, timesteps, weighting, noisy_latents = self.get_noise_pred_and_target(
                        args,
                        accelerator,
                        noise_scheduler,
                        latents,
                        batch,
                        text_encoder_conds,
                        unet,
                        network,
                        weight_dtype,
                        not args.network_train_text_encoder_only,
                        timesteps,
                        False,
                    )

                    if noise_pred.dtype not in {torch.float32, torch.float64}:
                        noise_pred = noise_pred.float()

                    if target.dtype not in {torch.float32, torch.float64}:
                        target = target.float()

                    # Compute loss
                    loss = train_util.conditional_loss(noise_pred, target, "l2", "none", None)

                    #if weighting is not None:
                    #    loss = loss * weighting
                    #if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    #    loss = apply_masked_loss(loss, batch)
                    loss = loss.mean(dim=[1, 2, 3])

                    loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                    loss = loss * loss_weights

                    # min snr gamma, scale v pred loss like noise pred, v pred like loss, debiased estimation etc.
                    loss = self.post_process_loss(loss, args, timesteps, noise_scheduler, False)

                    loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし
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
                           network, 
                           tokenizers, 
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
                           train_text_encoder=True):
        if not train_util.calculate_val_loss_check(args,global_step,epoch_step,val_dataloader,train_dataloader):
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

        timesteps_list = ast.literal_eval(args.validation_timesteps)
              
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
                loss = self.process_val_batch(network, batch, tokenizers, tokenize_strategy, text_encoders, 
                                              text_encoding_strategy, unet, vae, noise_scheduler, vae_dtype, 
                                              weight_dtype, accelerator, args, timesteps_list=timesteps_list, 
                                              train_text_encoder=train_text_encoder)
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

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)

        if args.disable_cuda_reduced_precision_operations:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)  # will be removed after sample_image is refactored

        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # データセットを準備する
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
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
            val_dataset_group = None # placeholder until validation dataset supported for arbitrary
            
        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)
        if val_dataset_group is not None:
            val_ds_for_collator = val_dataset_group if args.max_data_loader_n_workers == 0 else None
            val_collator = train_util.collator_class(current_epoch, current_step, val_ds_for_collator)

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()  # dasaset needs to know the strategies explicitly
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
              
        self.assert_extra_args(args, train_dataset_group)
        if val_dataset_group is not None:
            self.assert_extra_args(args, val_dataset_group)

        # acceleratorを準備する
        logger.info("preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        # モデルを読み込む
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # 差分追加学習のためにモデルを読み込む
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

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

        # 学習を準備する
        if cache_latents or val_dataset_group is not None:
            vae.to(device=accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()

            if cache_latents:
                train_dataset_group.new_cache_latents(vae, accelerator)

            if val_dataset_group is not None:
                print("Caching validation latents...")
                val_dataset_group.new_cache_latents(vae, accelerator)

            if cache_latents:
                vae.to(device="cpu")
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

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
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
        network_has_multiplier = hasattr(network, "set_multiplier")

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

        # make backward compatibility for text_encoder_lr
        support_multiple_lrs = hasattr(network, "prepare_optimizer_params_with_multiple_te_lrs")
        if support_multiple_lrs:
            text_encoder_lr = args.text_encoder_lr
        else:
            # toml backward compatibility
            if args.text_encoder_lr is None or isinstance(args.text_encoder_lr, float) or isinstance(args.text_encoder_lr, int):
                text_encoder_lr = args.text_encoder_lr
            else:
                text_encoder_lr = None if len(args.text_encoder_lr) == 0 else args.text_encoder_lr[0]
        
        try:
            if support_multiple_lrs:
                results = network.prepare_optimizer_params_with_multiple_te_lrs(text_encoder_lr, args.unet_lr, args.learning_rate)
            else:
                results = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr, args.learning_rate)
            if type(results) is tuple:
                trainable_params = results[0]
                lr_descriptions = results[1]
            else:
                trainable_params = results
                lr_descriptions = None
        except TypeError as e:
            trainable_params = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr)
            lr_descriptions = None

        # if len(trainable_params) == 0:
        #     accelerator.print("no trainable parameters found / 学習可能なパラメータが見つかりませんでした")
        # for params in trainable_params:
        #     for k, v in params.items():
        #         if type(v) == float:
        #             pass
        #         else:
        #             v = len(v)
        #         accelerator.print(f"trainable_params: {k} = {v}")

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)
        optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

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
            pin_memory=args.pin_data_loader_memory or args.pin_memory,
            persistent_workers=args.persistent_data_loader_workers,
        )
        
        if val_dataset_group is not None:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset_group,
                shuffle=False,
                batch_size=1,
                collate_fn=val_collator,
                num_workers=n_workers,
                pin_memory=args.pin_data_loader_memory or args.pin_memory,
                persistent_workers=args.persistent_data_loader_workers,
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
            network.to(dtype=weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(dtype=weight_dtype)

        if bool(args.train_network_norm_modules_as_float32) if args.train_network_norm_modules_as_float32 else False:
            # Recast normalization layers and their children back to FP32
            train_util.convert_named_modules_to_fp32(network)

        if args.conv2d_padding_mode is not None and args.conv2d_padding_mode.lower() != 'zeros':
            train_util.set_padding_mode_for_conv2d_modules(network, args.conv2d_padding_mode)

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
            unet.to(dtype=unet_weight_dtype)  # do not move to device because unet is not prepared by accelerator

        unet.requires_grad_(False)
        unet.to(dtype=unet_weight_dtype)
        for i, t_enc in enumerate(text_encoders):
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
            if t_enc.device.type != "cpu":
                t_enc.to(dtype=te_weight_dtype)

                # nn.Embedding not support FP8
                if te_weight_dtype != weight_dtype:
                    self.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)

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
                unet.to(device=accelerator.device, dtype=unet_weight_dtype)  # move to device because unet is not prepared by accelerator
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
            for i, (t_enc, flag) in enumerate(zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))):
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if flag:
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
            vae.to(device=accelerator.device, dtype=vae_dtype)

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

            # save current ecpoch and step
            train_state_file = os.path.join(output_dir, "train_state.json")
            # +1 is needed because the state is saved before current_step is set from global_step
            logger.info(f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value+1}")
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

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
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "random_crop_padding_percent": float(subset.random_crop_padding_percent),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "shuffle_caption_sigma": subset.shuffle_caption_sigma,
                        "keep_tokens": subset.keep_tokens,
                        "keep_tokens_separator": subset.keep_tokens_separator,
                        "secondary_separator": subset.secondary_separator,
                        "enable_wildcard": bool(subset.enable_wildcard),
                        "caption_prefix": subset.caption_prefix,
                        "caption_suffix": subset.caption_suffix,
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
                    "ss_random_crop_padding_percent": float(args.random_crop_padding_percent),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_shuffle_caption_sigma": args.shuffle_caption_sigma,
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_val_dataset_dirs": json.dumps(val_dataset_dirs_info),
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
                # num steps per epoch is calculated by num_processes and gradient_accumulation_steps
                initial_step = (args.initial_epoch - 1) * math.ceil(
                    len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
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

        progress_bar = tqdm(
            range(args.max_train_steps - initial_step), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps"
        )

        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
                if not args.resume:
                    logger.info(
                        f"initial_step is specified but not resuming. lr scheduler will be started from the beginning / initial_stepが指定されていますがresumeしていないため、lr schedulerは最初から始まります"
                    )
                logger.info(f"skipping {initial_step} steps / {initial_step}ステップをスキップします")
                initial_step *= args.gradient_accumulation_steps

                # set epoch to start to make initial_step less than len(train_dataloader)
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            else:
                # if not, only epoch no is skipped for informative purpose
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                initial_step = 0  # do not skip

        global_step = 0

        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

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
                                                                      device=accelerator.device,
                                                                      dtype=torch.float64 if args.edm2_loss_weighting_use_float64 else torch.float32)
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
                train_util.plot_dynamic_loss_weighting(args, 0, lossweightMLP, 1000, accelerator.device)

            if args.edm2_loss_weighting_laplace:
                train_util.calculate_edm2_laplace(lossweightMLP, noise_scheduler, accelerator.device)
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
                "network_train" if args.log_tracker_name is None else args.log_tracker_name,
                config=train_util.get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        loss_recorder = train_util.LossRecorder()
        val_loss_recorder = train_util.LossRecorder()

        if args.edm2_loss_weighting:
            loss_scaled_recorder = train_util.LossRecorder()
        
        del train_dataset_group

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
        if self.is_text_encoder_not_needed_for_training(args):
            logger.info("text_encoder is not needed for training. deleting to save memory.")
            for t_enc in text_encoders:
                del t_enc
            text_encoders = []
            text_encoder = None

        grad_norm = 0.0
        edm2_grad_norm = 0.0
        grad_norm_clipped = 0.0
        edm2_grad_norm_clipped = 0.0
        current_val_loss, average_val_loss, val_logs = None, None, {}
        keys_scaled, mean_norm, maximum_norm = None, None, None
        max_mean_logs = {}
        current_global_step_loss = 0.0
        current_global_step_loss_scaled = 0.0 if args.edm2_loss_weighting else None
        average_loss_scaled = 0.0 if args.edm2_loss_weighting else None
        avr_loss = 0.0
        gradient_stats = {
                'train/grad_norm/mean': 0.0,
                'train/grad_norm/median': 0.0,
                'train/grad_norm/std': 0.0,
                'train/grad_norm/min': 0.0,
                'train/grad_norm/max': 0.0,
                'train/grad_norm/p10': 0.0,
                'train/grad_norm/p25': 0.0,
                'train/grad_norm/p75': 0.0,
                'train/grad_norm/p90': 0.0,
                'train/grad_norm/p95': 0.0,
                'train/grad_norm/p98': 0.0,
                'train/grad_norm/p99': 0.0,
                'train/grad_norm/p995': 0.0,
                'train/grad_norm/p998': 0.0,
                'train/grad_norm/p999': 0.0,
            }
        network_norm_stats = {
                'model/module_norm/mean': 0.0,
                'model/module_norm/median': 0.0,
                'model/module_norm/std': 0.0,
                'model/module_norm/min': 0.0,
                'model/module_norm/max': 0.0,
                'model/module_norm/p10': 0.0,
                'model/module_norm/p25': 0.0,
                'model/module_norm/p75': 0.0,
                'model/module_norm/p90': 0.0,
                'model/module_norm/p95': 0.0,
                'model/module_norm/p98': 0.0,
                'model/module_norm/p99': 0.0,
                'model/module_norm/p995': 0.0,
                'model/module_norm/p998': 0.0,
                'model/module_norm/p999': 0.0,
                'model/module_norm/unscaled/mean': 0.0,
                'model/module_norm/unscaled/median': 0.0,
                'model/module_norm/unscaled/std': 0.0,
                'model/module_norm/unscaled/min': 0.0,
                'model/module_norm/unscaled/max': 0.0,
                'model/module_norm/unscaled/p10': 0.0,
                'model/module_norm/unscaled/p25': 0.0,
                'model/module_norm/unscaled/p75': 0.0,
                'model/module_norm/unscaled/p90': 0.0,
                'model/module_norm/unscaled/p95': 0.0,
                'model/module_norm/unscaled/p98': 0.0,
                'model/module_norm/unscaled/p99': 0.0,
                'model/module_norm/unscaled/p995': 0.0,
                'model/module_norm/unscaled/p998': 0.0,
                'model/module_norm/unscaled/p999': 0.0,
            }

        # For --sample_at_first
        if train_util.sample_images_check(args, 0, global_step) or train_util.calculate_val_loss_check(args, global_step, 0, val_dataloader, train_dataloader):
            #Switch network to eval mode
            network.eval()
            optimizer_eval_fn()
            self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
            if train_util.calculate_val_loss_check(args, global_step, 0, val_dataloader, train_dataloader):
                current_val_loss, average_val_loss, val_logs = self.calculate_val_loss(global_step, 0, train_dataloader, val_loss_recorder, val_dataloader, cyclic_val_dataloader, network, tokenizers, tokenize_strategy, text_encoders, text_encoding_strategy, unet, vae, noise_scheduler, vae_dtype, weight_dtype, accelerator, args, train_text_encoder)
            #Switch network to train mode
            optimizer_train_fn()
            network.train()

        if len(accelerator.trackers) > 0:
            logs = self.generate_step_logs(
                args=args, 
                current_loss=current_global_step_loss, 
                avr_loss=avr_loss, 
                lr_scheduler=lr_scheduler, 
                lr_descriptions=lr_descriptions, 
                optimizer=optimizer, 
                keys_scaled=keys_scaled, 
                mean_norm=mean_norm, 
                maximum_norm=maximum_norm, 
                grad_norm=grad_norm, 
                grad_norm_clipped=grad_norm_clipped, 
                current_val_loss=current_val_loss, 
                average_val_loss=average_val_loss, 
                current_loss_scaled=current_global_step_loss_scaled, 
                average_loss_scaled=average_loss_scaled, 
                edm2_grad_norm=edm2_grad_norm, 
                edm2_grad_norm_clipped=edm2_grad_norm_clipped, 
                edm2_lr_scheduler=mlp_lr_scheduler, 
                gradient_stats=gradient_stats, 
                network_norm_stats=network_norm_stats
            )
            accelerator.log(logs, step=0)

        # training loop
        if initial_step > 0:  # only if skip_until_initial_step is specified
            for skip_epoch in range(epoch_to_start):  # skip epochs
                logger.info(f"skipping epoch {skip_epoch+1} because initial_step (multiplied) is {initial_step}")
                initial_step -= len(train_dataloader)
            global_step = initial_step

        # log device and dtype for each model
        logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
        for i, t_enc in enumerate(text_encoders):
            params_itr = t_enc.parameters()
            params_itr.__next__()  # skip the first parameter
            params_itr.__next__()  # skip the second parameter. because CLIP first two parameters are embeddings
            param_3rd = params_itr.__next__()
            logger.info(f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}")
            
        clean_memory_on_device(accelerator.device)

        # Define the number of steps to accumulate gradients
        iter_size = args.gradient_accumulation_steps
        accumulation_counter = 0

        if args.grokfast_type:
            if args.grokfast_type.lower() == "ema":
                grad_filter = Gradfilter_ema(accelerator.unwrap_model(network), 
                                             alpha=float(args.grokfast_ema_alpha) if args.grokfast_ema_alpha is not None else 0.98, 
                                             lamb=float(args.grokfast_lamb) if args.grokfast_lamb is not None else 2.0, 
                                             warmup_steps=int(args.grokfast_warmup_steps) if args.grokfast_warmup_steps is not None else 0, 
                                             dtype=weight_dtype)
            elif args.grokfast_type.lower() == "ma":
                gf_window_size = int(args.grokfast_ma_window_size) if args.grokfast_ma_window_size is not None else 25

                grad_filter = Gradfilter_ma(accelerator.unwrap_model(network), 
                                            window_size=gf_window_size,
                                            lamb=float(args.grokfast_lamb) if args.grokfast_lamb is not None else 5.0, 
                                            filter_type=args.grokfast_ma_filter_type, 
                                            warmup_steps=int(args.grokfast_warmup_steps) if args.grokfast_warmup_steps is not None else gf_window_size, 
                                            dtype=weight_dtype)
            accelerator.register_for_checkpointing(grad_filter)

        if args.sangoi_loss_modifier:
            if args.zero_terminal_snr:
                logger.warning("As zero terminal SNR is set, setting min snr for sangoi loss modifier to zero.")
            if args.min_snr_gamma:
                logger.warning("Min snr gamma and sangoi loss modification both limit the max snr, ignoring min snr gamma in favor of sangoi.")

        if args.full_bf16:
            # apply stochastic grad accumulator hooks
            stochastic_accumulator.StochasticAccumulator.assign_hooks(network)

            for epoch in range(epoch_to_start, num_train_epochs):
                accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
                current_epoch.value = epoch + 1

                metadata["ss_epoch"] = str(epoch + 1)

                accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)

                skipped_dataloader = None
                if initial_step > 0:
                    skipped_dataloader = accelerator.skip_first_batches(train_dataloader, initial_step - 1)
                    initial_step = 1

                for step, batch in enumerate(skipped_dataloader or train_dataloader):
                    current_step.value = global_step
                    if initial_step > 0:
                        initial_step -= 1
                        continue

                    # Determine whether we should synchronize gradients
                    sync_gradients = (accumulation_counter + 1) % iter_size == 0 or (step + 1 == len(skipped_dataloader or train_dataloader))

                    effective_batch_size = accumulation_counter + 1 if sync_gradients else iter_size
                    grad_accum_loss_scaling = 1.0 / effective_batch_size

                    # Prevent gradient synchronization during accumulation steps in distributed settings
                    with self.determine_grad_sync_context(accelerator, sync_gradients, training_model, lossweightMLP):
                        # Perform forward and backward passes
                        # Processing batch and computing loss

                        # Call any required functions at the start of the step
                        on_step_start_for_network(text_encoder, unet)

                        # Temporary, for batch processing
                        self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype)

                        # Prepare latents
                        if "latents" in batch and batch["latents"] is not None:
                            latents = batch["latents"].to(device=accelerator.device, dtype=weight_dtype)
                        else:
                            with torch.no_grad():
                                # Convert images to latents
                                latents = self.encode_images_to_latents(args, accelerator, vae, batch["images"].to(device=vae.device, dtype=vae_dtype))
                                latents = latents.to(dtype=weight_dtype)
                                # Replace NaNs if any
                                if torch.any(torch.isnan(latents)):
                                    accelerator.print("NaN found in latents, replacing with zeros")
                                    latents = torch.nan_to_num(latents, 0, out=latents)

                        latents = self.shift_scale_latents(args, latents)

                        # Handle network multipliers
                        if network_has_multiplier:
                            multipliers = batch["network_multipliers"]
                            if torch.all(multipliers == multipliers[0]):
                                multipliers = multipliers[0].item()
                            else:
                                raise NotImplementedError("Multipliers for each sample are not supported yet")
                            accelerator.unwrap_model(network).set_multiplier(multipliers)

                        # Prepare text encoder conditions
                        text_encoder_conds = batch.get("text_encoder_outputs_list", [])
                        if not text_encoder_conds or text_encoder_conds[0] is None or train_text_encoder:
                            with torch.set_grad_enabled(train_text_encoder), accelerator.autocast():
                                # Get the text embeddings
                                if args.weighted_captions:
                                    input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                                        tokenize_strategy,
                                        self.get_models_for_text_encoding(args, accelerator, text_encoders),
                                        input_ids_list,
                                        weights_list,
                                    )
                                else:
                                    input_ids = [ids.to(device=accelerator.device) for ids in batch["input_ids_list"]]
                                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                                        tokenize_strategy,
                                        self.get_models_for_text_encoding(args, accelerator, text_encoders),
                                        input_ids,
                                    )
                                if args.full_fp16:
                                    encoded_text_encoder_conds = [c.to(dtype=weight_dtype) for c in encoded_text_encoder_conds]
                            # Update text encoder conditions
                            if not text_encoder_conds:
                                text_encoder_conds = encoded_text_encoder_conds
                            else:
                                for i in range(len(encoded_text_encoder_conds)):
                                    if encoded_text_encoder_conds[i] is not None:
                                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

                        # Get noise prediction and target
                        noise_pred, target, timesteps, weighting, noisy_latents = self.get_noise_pred_and_target(
                            args,
                            accelerator,
                            noise_scheduler,
                            latents,
                            batch,
                            text_encoder_conds,
                            unet,
                            network,
                            weight_dtype,
                            train_unet,
                        )

                        if noise_pred.dtype not in {torch.float32, torch.float64}:
                            noise_pred = noise_pred.float()

                        if target.dtype not in {torch.float32, torch.float64}:
                            target = target.float()

                        huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                        # Compute loss
                        loss = train_util.conditional_loss(noise_pred, target, args.loss_type, "none", huber_c, scale=float(args.loss_scale))

                        if weighting is not None:
                            loss = loss * weighting
                        if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                            loss = apply_masked_loss(loss, batch)
                        loss = loss.mean(dim=[1, 2, 3])  # Mean over dimensions

                        loss_weights = batch["loss_weights"]  # Sample-wise weights
                        loss = loss * loss_weights

                        if args.sangoi_loss_modifier:
                            # Min SNR should be zero for zero_terminal_snr
                            if args.zero_terminal_snr:
                                min_snr = 0
                            else:
                                min_snr = float(args.sangoi_loss_modifier_min_snr)

                            loss = loss * train_util.sangoi_loss_modifier(timesteps, 
                                                                    noise_pred, 
                                                                    target, 
                                                                    noise_scheduler,
                                                                    min_snr,
                                                                    float(args.sangoi_loss_modifier_max_snr))

                        # min snr gamma, scale v pred loss like noise pred, v pred like loss, debiased estimation etc.
                        loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

                        if args.loss_multipler or args.loss_multiplier:
                            loss.mul_(float(args.loss_multipler or args.loss_multiplier) if args.loss_multipler is not None or args.loss_multiplier is not None else 1.0)

                        # For logging
                        pre_scaling_loss = loss.mean()

                        if args.edm2_loss_weighting:
                            loss, loss_scaled = lossweightMLP(loss, timesteps)
                            loss_scaled = loss_scaled.mean()
                            loss_scaled = loss_scaled * grad_accum_loss_scaling

                        loss = loss.mean()  # Mean over batch

                        # Divide loss by iter_size to average over accumulated steps
                        loss = loss * grad_accum_loss_scaling

                        # Backward pass
                        accelerator.backward(loss)

                        # Replace loss with pre_scaling_loss, scaled by grad_accum_loss_scaling
                        loss = pre_scaling_loss * grad_accum_loss_scaling

                        accumulation_counter += 1

                    current_global_step_loss += loss.detach().item() / grad_accum_loss_scaling
                    if args.edm2_loss_weighting:
                        current_global_step_loss_scaled += loss_scaled.detach().item() / grad_accum_loss_scaling
                    else:
                        current_global_step_loss_scaled = None

                    if sync_gradients:
                        # apply grad buffer back
                        stochastic_accumulator.StochasticAccumulator.reassign_grad_buffer(network)

                        self.all_reduce_network(accelerator, network)  # sync DDP grad manually

                        params_to_analyze = accelerator.unwrap_model(network).get_trainable_params()
                        gradient_stats = analyze_gradient_norms(params_to_analyze)

                        params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                        if args.max_grad_norm != 0.0:
                            grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm).item()
                            grad_norm_clipped = min(grad_norm, args.max_grad_norm)
                        else: 
                            grad_norm = accelerator.clip_grad_norm_(params_to_clip, float('inf')).item()
                            grad_norm_clipped = grad_norm

                        if args.grokfast_type:
                            grad_filter.filter()

                        # Perform step
                        optimizer.step()

                        # Zero gradients
                        optimizer.zero_grad(set_to_none=True)

                        if args.edm2_loss_weighting:
                            self.all_reduce_network(accelerator, lossweightMLP)  # sync DDP grad manually
                            params_to_clip = accelerator.unwrap_model(lossweightMLP).get_trainable_params()
                            edm2_loss_weighting_max_grad_norm = float(args.edm2_loss_weighting_max_grad_norm) if args.edm2_loss_weighting_max_grad_norm is not None else 1.0
                            if edm2_loss_weighting_max_grad_norm != 0.0:
                                edm2_grad_norm = accelerator.clip_grad_norm_(params_to_clip, edm2_loss_weighting_max_grad_norm).item()
                                edm2_grad_norm_clipped = min(edm2_grad_norm, edm2_loss_weighting_max_grad_norm)
                            else: 
                                edm2_grad_norm = accelerator.clip_grad_norm_(params_to_clip, float('inf')).item()
                                edm2_grad_norm_clipped = edm2_grad_norm

                            MLP_optim.step()

                        # Zero gradients
                        optimizer.zero_grad(set_to_none=True)

                        if args.edm2_loss_weighting:
                            MLP_optim.zero_grad(set_to_none=True)

                        # Update learning rate
                        lr_scheduler.step()

                        if args.edm2_loss_weighting and mlp_lr_scheduler is not None:
                            mlp_lr_scheduler.step()

                        if args.scale_weight_norms:
                            keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                                args.scale_weight_norms, accelerator.device
                            )

                            if isinstance(keys_scaled, torch.Tensor):
                                #Unpack
                                keys_scaled = keys_scaled.item()

                            if isinstance(mean_norm, torch.Tensor):
                                #Unpack
                                mean_norm = mean_norm.item()

                            if isinstance(maximum_norm, torch.Tensor):
                                #Unpack
                                maximum_norm = maximum_norm.item()

                            max_mean_logs = {"Keys Scaled": keys_scaled, "Avg key norm": mean_norm}
                        else:
                            keys_scaled, mean_norm, maximum_norm = None, None, None

                        if hasattr(network, "get_norms"):
                            unscaled_norms, scaled_norms = accelerator.unwrap_model(network).get_norms(accelerator.device)
                            network_norm_stats = analyze_model_norms(unscaled_norms, scaled_norms)

                        progress_bar.update(1)
                        global_step += 1

                        if self.plot_dynamic_loss_weighting_check(args, global_step):
                            train_util.plot_dynamic_loss_weighting(args, global_step, lossweightMLP, 1000, accelerator.device)

                        if args.edm2_loss_weighting and args.edm2_loss_weighting_laplace:
                            train_util.calculate_edm2_laplace(lossweightMLP, noise_scheduler, accelerator.device)

                        if (train_util.sample_images_check(args, None, global_step) or 
                            train_util.calculate_val_loss_check(args, global_step, step, val_dataloader, train_dataloader) or 
                            args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0):
                            #Switch network to train mode
                            network.eval()
                            optimizer_eval_fn()                        
                            self.sample_images(
                                accelerator, args, None, global_step, accelerator.device, vae, tokenizers, text_encoder, unet
                            )

                            if train_util.calculate_val_loss_check(args, global_step, step, val_dataloader, train_dataloader):
                                current_val_loss, average_val_loss, val_logs = self.calculate_val_loss(global_step, step, skipped_dataloader or train_dataloader, val_loss_recorder, val_dataloader, cyclic_val_dataloader, network, tokenizers, tokenize_strategy, text_encoders, text_encoding_strategy, unet, vae, noise_scheduler, vae_dtype, weight_dtype, accelerator, args, train_text_encoder)
                            else:
                                current_val_loss, average_val_loss, val_logs = None, None, None

                            # 指定ステップごとにモデルを保存
                            if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                                accelerator.wait_for_everyone()
                                if accelerator.is_main_process:
                                    ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                                    if args.edm2_loss_weighting:
                                        loss_weights_ckpt_name = train_util.get_step_loss_weights_ckpt_name(args, "." + args.save_model_as, global_step)
                                        save_model(loss_weights_ckpt_name, accelerator.unwrap_model(lossweightMLP), global_step, epoch, dtype_override=torch.float64)

                                    if args.save_state:
                                        train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                                    remove_step_no = train_util.get_remove_step_no(args, global_step)
                                    if remove_step_no is not None:
                                        remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                        remove_model(remove_ckpt_name)

                                        if args.edm2_loss_weighting:
                                            remove_loss_weights_ckpt_name = train_util.get_step_loss_weights_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                            remove_model(remove_loss_weights_ckpt_name)
                            #Switch network to train mode
                            optimizer_train_fn()
                            network.train()
                        else:
                            current_val_loss, average_val_loss, val_logs = None, None, None

                        loss_recorder.add(epoch=epoch, step=global_step, loss=current_global_step_loss / accumulation_counter)
                        if args.edm2_loss_weighting:
                            loss_scaled_recorder.add(epoch=epoch, step=global_step, loss=current_global_step_loss_scaled / accumulation_counter)

                        avr_loss: float = loss_recorder.moving_average if global_step > 0 else 0.0
                        logs = {"avg_loss": avr_loss}

                        if args.scale_weight_norms:
                            logs = {**max_mean_logs, **logs}

                        progress_bar.set_postfix(**logs)

                        if len(accelerator.trackers) > 0:
                            current_global_step_loss = (current_global_step_loss / accumulation_counter)
                            if args.edm2_loss_weighting:
                                current_global_step_loss_scaled = (current_global_step_loss_scaled / accumulation_counter)
                                average_loss_scaled: float = loss_scaled_recorder.moving_average
                            else:
                                current_global_step_loss_scaled = None
                                average_loss_scaled = None
                            logs = self.generate_step_logs(
                                args, current_global_step_loss, avr_loss, lr_scheduler, lr_descriptions, 
                                optimizer, keys_scaled, mean_norm, maximum_norm, grad_norm, 
                                grad_norm_clipped, current_val_loss, average_val_loss, current_global_step_loss_scaled, 
                                average_loss_scaled, edm2_grad_norm, edm2_grad_norm_clipped, mlp_lr_scheduler, gradient_stats, network_norm_stats
                            )
                            accelerator.log(logs, step=global_step)
                            current_global_step_loss = 0.0
                            if args.edm2_loss_weighting:
                                current_global_step_loss_scaled = 0.0

                        # Reset accumulation counter
                        accumulation_counter = 0

                    if global_step >= args.max_train_steps:
                        break

                if len(accelerator.trackers) > 0:
                    logs = {"loss/epoch": loss_recorder.moving_average}
                    accelerator.log(logs, step=global_step)
                            
                accelerator.wait_for_everyone()

                if (train_util.sample_images_check(args, epoch + 1, global_step) or 
                    args.save_every_n_epochs is not None):
                    # 指定エポックごとにモデルを保存
                    #Switch network to eval mode
                    network.eval()
                    optimizer_eval_fn()
                    if args.save_every_n_epochs is not None:
                        saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                        if is_main_process and saving:
                            ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                            if args.edm2_loss_weighting:
                                loss_weights_ckpt_name = train_util.get_epoch_loss_weights_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                                save_model(loss_weights_ckpt_name, accelerator.unwrap_model(lossweightMLP), global_step, epoch + 1, dtype_override=torch.float64)

                            remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                            if remove_epoch_no is not None:
                                remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                                remove_model(remove_ckpt_name)

                                if args.edm2_loss_weighting:
                                    remove_loss_weights_ckpt_name = train_util.get_epoch_loss_weights_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                                    remove_model(remove_loss_weights_ckpt_name)

                            if args.save_state:
                                train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)
                    
                    self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
                    #Switch network to train mode
                    optimizer_train_fn()
                    network.train()

                # end of epoch

        else:
            #Normal training loop
            for epoch in range(epoch_to_start, num_train_epochs):
                accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
                current_epoch.value = epoch + 1

                metadata["ss_epoch"] = str(epoch + 1)

                accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)

                skipped_dataloader = None
                if initial_step > 0:
                    skipped_dataloader = accelerator.skip_first_batches(train_dataloader, initial_step - 1)
                    initial_step = 1


                for step, batch in enumerate(skipped_dataloader or train_dataloader):
                    current_step.value = global_step
                    if initial_step > 0:
                        initial_step -= 1
                        continue

                    with accelerator.accumulate(training_model, lossweightMLP) if args.edm2_loss_weighting else accelerator.accumulate(training_model):
                        on_step_start_for_network(text_encoder, unet)

                        accumulation_counter += 1

                        # temporary, for batch processing
                        self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype)

                        if "latents" in batch and batch["latents"] is not None:
                            latents = batch["latents"].to(device=accelerator.device, dtype=weight_dtype)
                        else:
                            with torch.no_grad():
                                # latentに変換
                                latents = self.encode_images_to_latents(args, accelerator, vae, batch["images"].to(device=vae.device, dtype=vae_dtype))
                                latents = latents.to(dtype=weight_dtype)

                                # NaNが含まれていれば警告を表示し0に置き換える
                                if torch.any(torch.isnan(latents)):
                                    accelerator.print("NaN found in latents, replacing with zeros")
                                    latents = torch.nan_to_num(latents, 0, out=latents)

                        latents = self.shift_scale_latents(args, latents)

                        # get multiplier for each sample
                        if network_has_multiplier:
                            multipliers = batch["network_multipliers"]
                            # if all multipliers are same, use single multiplier
                            if torch.all(multipliers == multipliers[0]):
                                multipliers = multipliers[0].item()
                            else:
                                raise NotImplementedError("multipliers for each sample is not supported yet")
                            # print(f"set multiplier: {multipliers}")
                            accelerator.unwrap_model(network).set_multiplier(multipliers)

                        text_encoder_conds = []
                        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                        if text_encoder_outputs_list is not None:
                            text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs

                        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
                            # TODO this does not work if 'some text_encoders are trained' and 'some are not and not cached'
                            with torch.set_grad_enabled(train_text_encoder), accelerator.autocast():
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
                                    input_ids = [ids.to(device=accelerator.device) for ids in batch["input_ids_list"]]
                                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                                        tokenize_strategy,
                                        self.get_models_for_text_encoding(args, accelerator, text_encoders),
                                        input_ids,
                                    )
                                if args.full_fp16:
                                    encoded_text_encoder_conds = [c.to(dtype=weight_dtype) for c in encoded_text_encoder_conds]

                            # if text_encoder_conds is not cached, use encoded_text_encoder_conds
                            if len(text_encoder_conds) == 0:
                                text_encoder_conds = encoded_text_encoder_conds
                            else:
                                # if encoded_text_encoder_conds is not None, update cached text_encoder_conds
                                for i in range(len(encoded_text_encoder_conds)):
                                    if encoded_text_encoder_conds[i] is not None:
                                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

                        # Get noise prediction and target
                        noise_pred, target, timesteps, weighting, noisy_latents = self.get_noise_pred_and_target(
                            args,
                            accelerator,
                            noise_scheduler,
                            latents,
                            batch,
                            text_encoder_conds,
                            unet,
                            network,
                            weight_dtype,
                            train_unet,
                        )

                        if noise_pred.dtype not in {torch.float32, torch.float64}:
                            noise_pred = noise_pred.float()

                        if target.dtype not in {torch.float32, torch.float64}:
                            target = target.float()

                        huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                        # Compute loss
                        loss = train_util.conditional_loss(noise_pred, target, args.loss_type, "none", huber_c, scale=float(args.loss_scale))

                        if weighting is not None:
                            loss = loss * weighting
                        if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                            loss = apply_masked_loss(loss, batch)
                        loss = loss.mean(dim=[1, 2, 3])

                        loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                        loss = loss * loss_weights

                        if args.sangoi_loss_modifier:
                            # Min SNR should be zero for zero_terminal_snr
                            if args.zero_terminal_snr:
                                min_snr = 0
                            else:
                                min_snr = float(args.sangoi_loss_modifier_min_snr)

                            loss = loss * train_util.sangoi_loss_modifier(timesteps, 
                                                                    noise_pred, 
                                                                    target, 
                                                                    noise_scheduler,
                                                                    min_snr,
                                                                    float(args.sangoi_loss_modifier_max_snr))

                        # min snr gamma, scale v pred loss like noise pred, v pred like loss, debiased estimation etc.
                        loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

                        if args.loss_multipler or args.loss_multiplier:
                            loss.mul_(float(args.loss_multipler or args.loss_multiplier) if args.loss_multipler is not None or args.loss_multiplier is not None else 1.0)

                        # For logging
                        pre_scaling_loss = loss.mean()

                        if args.edm2_loss_weighting:
                            loss, loss_scaled = lossweightMLP(loss, timesteps)
                            loss_scaled = loss_scaled.mean()

                        loss = loss.mean()  # Mean over batch
                        
                        accelerator.backward(loss)

                        loss = pre_scaling_loss

                        if accelerator.sync_gradients:
                            self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                            params_to_analyze = accelerator.unwrap_model(network).get_trainable_params()
                            gradient_stats = analyze_gradient_norms(params_to_analyze)

                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()

                            if args.max_grad_norm != 0.0:
                                grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm).item()
                                grad_norm_clipped = min(grad_norm, args.max_grad_norm)
                            else: 
                                grad_norm = accelerator.clip_grad_norm_(params_to_clip, float('inf')).item()
                                grad_norm_clipped = grad_norm

                            if args.grokfast_type:
                                grad_filter.filter()

                        optimizer.step()

                        if args.edm2_loss_weighting:
                            if accelerator.sync_gradients:
                                self.all_reduce_network(accelerator, lossweightMLP)  # sync DDP grad manually
                                params_to_clip = accelerator.unwrap_model(lossweightMLP).get_trainable_params()
                                edm2_loss_weighting_max_grad_norm = float(args.edm2_loss_weighting_max_grad_norm) if args.edm2_loss_weighting_max_grad_norm is not None else 1.0
                                if edm2_loss_weighting_max_grad_norm != 0.0:
                                    edm2_grad_norm = accelerator.clip_grad_norm_(params_to_clip, edm2_loss_weighting_max_grad_norm).item()
                                    edm2_grad_norm_clipped = min(edm2_grad_norm, edm2_loss_weighting_max_grad_norm)
                                else: 
                                    edm2_grad_norm = accelerator.clip_grad_norm_(params_to_clip, float('inf')).item()
                                    edm2_grad_norm_clipped = edm2_grad_norm

                            MLP_optim.step()

                        lr_scheduler.step()

                        if args.edm2_loss_weighting and mlp_lr_scheduler is not None:
                            mlp_lr_scheduler.step()

                        optimizer.zero_grad(set_to_none=True)

                        if args.edm2_loss_weighting:
                            MLP_optim.zero_grad(set_to_none=True)

                    # Should only scale weight norms AFTER an actual optimizer step, not unaccumulated steps
                    # thus should check if accelerator.sync_gradients is true
                    if args.scale_weight_norms and accelerator.sync_gradients:
                        keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                            args.scale_weight_norms, accelerator.device
                        )

                        if isinstance(keys_scaled, torch.Tensor):
                            #Unpack
                            keys_scaled = keys_scaled.item()

                        if isinstance(mean_norm, torch.Tensor):
                            #Unpack
                            mean_norm = mean_norm.item()

                        if isinstance(maximum_norm, torch.Tensor):
                            #Unpack
                            maximum_norm = maximum_norm.item()

                        max_mean_logs = {"Keys Scaled": keys_scaled, "Avg key norm": mean_norm}
                    else:
                        keys_scaled, mean_norm, maximum_norm = None, None, None

                    if accelerator.sync_gradients and hasattr(network, "get_norms"):
                        unscaled_norms, scaled_norms = accelerator.unwrap_model(network).get_norms(accelerator.device)
                        network_norm_stats = analyze_model_norms(unscaled_norms, scaled_norms)
                    else:
                        network_norm_stats = None

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        if self.plot_dynamic_loss_weighting_check(args, global_step):
                            train_util.plot_dynamic_loss_weighting(args, global_step, lossweightMLP, 1000, accelerator.device)

                        if args.edm2_loss_weighting and args.edm2_loss_weighting_laplace:
                            train_util.calculate_edm2_laplace(lossweightMLP, noise_scheduler, accelerator.device)

                        if (train_util.sample_images_check(args, None, global_step) or 
                            train_util.calculate_val_loss_check(args, global_step, step, val_dataloader, train_dataloader) or 
                            args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0):
                            #Switch network to eval mode
                            network.eval()
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
                                                                                                       network, tokenizers, 
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
                                                                                                       train_text_encoder)
                            else:
                                current_val_loss, average_val_loss, val_logs = None, None, None

                            # 指定ステップごとにモデルを保存
                            if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                                accelerator.wait_for_everyone()
                                if accelerator.is_main_process:
                                    ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                                    if args.edm2_loss_weighting:
                                        loss_weights_ckpt_name = train_util.get_step_loss_weights_ckpt_name(args, "." + args.save_model_as, global_step)
                                        save_model(loss_weights_ckpt_name, accelerator.unwrap_model(lossweightMLP), global_step, epoch, dtype_override=torch.float64)

                                    if args.save_state:
                                        train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                                    remove_step_no = train_util.get_remove_step_no(args, global_step)
                                    if remove_step_no is not None:
                                        remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                        remove_model(remove_ckpt_name)

                                        if args.edm2_loss_weighting:
                                            remove_loss_weights_ckpt_name = train_util.get_step_loss_weights_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                            remove_model(remove_loss_weights_ckpt_name)

                            #Switch network to train mode
                            optimizer_train_fn()
                            network.train()
                        else:
                            current_val_loss, average_val_loss, val_logs = None, None, None

                    
                    current_global_step_loss += loss.detach().item()
                    if args.edm2_loss_weighting:
                        current_global_step_loss_scaled += loss_scaled.detach().item()
                    else:
                        current_global_step_loss_scaled = None

                    if accelerator.sync_gradients:
                        loss_recorder.add(epoch=epoch, step=global_step, loss=current_global_step_loss / accumulation_counter)
                        if args.edm2_loss_weighting:
                            loss_scaled_recorder.add(epoch=epoch, step=global_step, loss=current_global_step_loss_scaled / accumulation_counter)

                        avr_loss: float = loss_recorder.moving_average if global_step > 0 else 0.0
                        logs = {"avg_loss": avr_loss}

                        if args.scale_weight_norms:
                            logs = {**max_mean_logs, **logs}

                        progress_bar.set_postfix(**logs)

                        if len(accelerator.trackers) > 0:
                            current_global_step_loss = (current_global_step_loss / accumulation_counter)
                            if args.edm2_loss_weighting:
                                current_global_step_loss_scaled = (current_global_step_loss_scaled / accumulation_counter)
                                average_loss_scaled: float = loss_scaled_recorder.moving_average
                            else:
                                current_global_step_loss_scaled = None
                                average_loss_scaled = None
                            logs = self.generate_step_logs(
                                args, current_global_step_loss, avr_loss, lr_scheduler, lr_descriptions, 
                                optimizer, keys_scaled, mean_norm, maximum_norm, grad_norm, 
                                grad_norm_clipped, current_val_loss, average_val_loss, current_global_step_loss_scaled, 
                                average_loss_scaled, edm2_grad_norm, edm2_grad_norm_clipped, mlp_lr_scheduler, gradient_stats, network_norm_stats
                            )
                            accelerator.log(logs, step=global_step)
                            current_global_step_loss = 0.0
                            if args.edm2_loss_weighting:
                                current_global_step_loss_scaled = 0.0

                            accumulation_counter = 0
                                            
                    if global_step >= args.max_train_steps:
                        break

                if len(accelerator.trackers) > 0:
                    logs = {"loss/epoch": loss_recorder.moving_average}
                    accelerator.log(logs, step=global_step)
                                
                accelerator.wait_for_everyone()

                if (train_util.sample_images_check(args, epoch + 1, global_step) or 
                    args.save_every_n_epochs is not None):
                    # 指定エポックごとにモデルを保存
                    #Switch network to eval mode
                    network.eval()
                    optimizer_eval_fn()

                    if args.save_every_n_epochs is not None:
                        saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                        if is_main_process and saving:
                            ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                            if args.edm2_loss_weighting:
                                loss_weights_ckpt_name = train_util.get_epoch_loss_weights_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                                save_model(loss_weights_ckpt_name, accelerator.unwrap_model(lossweightMLP), global_step, epoch + 1, dtype_override=torch.float64)

                            remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                            if remove_epoch_no is not None:
                                remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                                remove_model(remove_ckpt_name)

                                if args.edm2_loss_weighting:
                                    remove_loss_weights_ckpt_name = train_util.get_epoch_loss_weights_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                                    remove_model(remove_loss_weights_ckpt_name)

                            if args.save_state:
                                train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)
                    
                    self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
                    #Switch network to train mode
                    optimizer_train_fn()
                    network.train()

                # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()
        #Switch network to eval mode
        network.eval()
        optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

            if args.edm2_loss_weighting:
                loss_weights_ckpt_name = train_util.get_last_loss_weights_ckpt_name(args, "." + args.save_model_as)
                save_model(loss_weights_ckpt_name, lossweightMLP, global_step, num_train_epochs, force_sync_upload=True, dtype_override=torch.float64)

            logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
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
        "--validation_timesteps",
        type=str,
        default=r"[10, 350, 500, 650, 990]",
        help="A list of timesteps to use for each validation step."
    )  

    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Number of max validation steps for counting validation loss. By default, validation will run entire validation dataset"
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
        "--grokfast_type",
        type=str,
        default=None,
        choices=[None, "ema", "ma"],
        help="Grokfast type to apply exponential moving average (ema), which stores one ema state per parameter, or moving average (ma) which stores up to the window size of gradients.",
    )
    parser.add_argument(
        "--grokfast_ema_alpha",
        type=float,
        default=0.98,
        help="Momentum hyperparameter of the EMA.",
    )
    parser.add_argument(
        "--grokfast_lamb",
        type=float,
        default=None,
        help="Amplifying factor hyperparameter of the filter. Default of 5.0 for moving average, 2.0 for exponential moving average.",
    )
    parser.add_argument(
        "--grokfast_warmup_steps",
        type=int,
        default=None,
        help="Number of steps to warmup, gradually increasing application of the filter.",
    )

    parser.add_argument(
        "--grokfast_ma_window_size",
        type=int,
        default=25,
        help="The width of the moving average filter window. additional memory requirements increases linearly with respect to the windows size.",
    )

    parser.add_argument(
        "--grokfast_ma_filter_type",
        type=str,
        default="mean",
        choices=[None, "mean", "sum"],
        help="Aggregation method for the moving average running queue.",
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
        "--train_network_norm_modules_as_float32",
        action="store_true",
        help="Trains network norm layers as float32, slight reduction in processing speed, possible minor benefit from greater precision.",
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
        "--conv2d_padding_mode",
        type=str,
        default='zeros',
        choices=["zeros", "reflect", "replicate","circular"],
        help="Adjusts the padding for edges of Conv2d modules, default is zeros, circular might have benefit, as it pads with the opposite side, tbd."
    )

    parser.add_argument(
        "--loss_related_use_float64",
        action="store_true",
        help="Upcasts targets, noise, noisy latents, latents, and loss during loss and noise calculations to float64 for greater precision. Slight compute and vram overhead."
    )

    parser.add_argument(
        "--edm2_loss_weighting_use_float64",
        action="store_true",
        help="Uses float64 for edm2 loss weighting."
    )

    parser.add_argument(
        "--disable_training_clip_l",
        action="store_true",
        help="Disable training clip l (first te), only effective if training TEs."
    )

    parser.add_argument(
        "--disable_training_clip_g",
        action="store_true",
        help="Disable training clip g (second te), only effective if training TEs."
    )

    # parser.add_argument("--loraplus_lr_ratio", default=None, type=float, help="LoRA+ learning rate ratio")
    # parser.add_argument("--loraplus_unet_lr_ratio", default=None, type=float, help="LoRA+ UNet learning rate ratio")
    # parser.add_argument("--loraplus_text_encoder_lr_ratio", default=None, type=float, help="LoRA+ text encoder learning rate ratio")
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)
