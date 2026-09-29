"""Stationary multi-scale random Fourier feature MMD loss primitives.

The loss compares each predicted-clean sample with a precomputed, immutable
reference embedding.  All stochastic features are supplied by the reference
cache; the per-step functions are deterministic and consume no global RNG.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from library import anchor_loss
from library.hf_token_loss import hf_snr_from_timesteps

ADV_EPS = 1e-6
DEFAULT_SIGMA_MULTIPLIERS = (0.5, 1.0, 2.0)


def validate_adv_args(
    scale: float,
    levels: int,
    features: int,
    tokens_per_band: int,
    reference_samples: int,
    bandwidths: str = "",
) -> None:
    if not math.isfinite(float(scale)):
        raise ValueError("adv_scale must be finite")
    if scale < 0.0:
        raise ValueError("adv_scale must be >= 0 (0 = off)")
    if levels < 1:
        raise ValueError("adv_levels must be >= 1")
    if features < 1:
        raise ValueError("adv_features must be >= 1")
    if tokens_per_band < 1:
        raise ValueError("adv_tokens_per_band must be >= 1")
    if reference_samples < 2:
        raise ValueError("adv_reference_samples must be >= 2")
    parse_bandwidths(bandwidths)


def parse_bandwidths(value: Optional[str]) -> Optional[List[float]]:
    """Parse an optional comma-separated, positive bandwidth ladder."""
    if value is None or not str(value).strip():
        return None
    try:
        sigmas = [float(part.strip()) for part in str(value).split(",")]
    except ValueError as exc:
        raise ValueError("adv_bandwidths must be comma-separated positive numbers") from exc
    if not sigmas or any(not math.isfinite(sigma) or sigma <= 0.0 for sigma in sigmas):
        raise ValueError("adv_bandwidths must contain only finite positive numbers")
    return sigmas


def _pool_grid(height: int, width: int, token_budget: int) -> tuple[int, int]:
    """Aspect-preserving adaptive-pool grid with at most ``token_budget`` cells."""
    if height * width <= token_budget:
        return height, width
    aspect = height / max(width, 1)
    grid_h = max(1, min(height, int(math.sqrt(token_budget * aspect))))
    grid_w = max(1, min(width, token_budget // grid_h))
    while grid_h * grid_w > token_budget:
        if grid_h / max(height, 1) > grid_w / max(width, 1) and grid_h > 1:
            grid_h -= 1
        elif grid_w > 1:
            grid_w -= 1
        else:
            break
    return grid_h, grid_w


def tokenize_band(band: torch.Tensor, token_budget: int) -> torch.Tensor:
    """Convert ``[B,C,H,W]`` to pooled ``[B,T,C]`` spatial value tokens."""
    if band.ndim != 4:
        raise ValueError(f"expected BCHW band, got shape {tuple(band.shape)}")
    grid = _pool_grid(band.shape[-2], band.shape[-1], token_budget)
    pooled = F.adaptive_avg_pool2d(band, grid)
    return pooled.flatten(2).transpose(1, 2)


def _rff_features(tokens: torch.Tensor, band_reference: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Evaluate the normalized multi-bandwidth RFF map in fp32."""
    x = tokens.float()
    mean = band_reference["mean"].to(device=x.device, dtype=torch.float32)
    std = band_reference["std"].to(device=x.device, dtype=torch.float32).clamp_min(ADV_EPS)
    omega = band_reference["omega"].to(device=x.device, dtype=torch.float32)
    phase = band_reference["phase"].to(device=x.device, dtype=torch.float32)
    x = (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    # [B,T,C] x [M,D,C] -> [B,M,T,D].
    projection = torch.einsum("btc,mdc->bmtd", x, omega) + phase.view(1, -1, 1, phase.shape[-1])
    count_features = omega.shape[1]
    count_kernels = omega.shape[0]
    features = math.sqrt(2.0 / count_features / count_kernels) * projection.cos()
    return features.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], count_kernels * count_features)


def adv_per_sample_loss(
    x0_pred: torch.Tensor,
    references: Dict[str, Dict[str, torch.Tensor]],
    levels: int,
    tokens_per_band: int,
) -> torch.Tensor:
    """Return the per-sample mean squared distance to fixed band embeddings.

    Args:
        x0_pred: predicted-clean BCHW tensor (gradient path retained).
        references: grid reference produced by :mod:`library.adv_cache`.
        levels: requested Laplacian levels, clamped identically to anchor loss.
        tokens_per_band: adaptive pooling budget used to build the references.

    Returns:
        fp32 tensor of shape ``[B]``. Values are non-negative and bounded by 2
        up to floating point roundoff for the normalized cosine features.
    """
    if x0_pred.ndim != 4:
        raise ValueError(f"adv loss requires a BCHW x0 prediction, got {tuple(x0_pred.shape)}")
    grid_key = f"{x0_pred.shape[-2]}x{x0_pred.shape[-1]}"
    if grid_key not in references:
        raise KeyError(f"no stationary adv reference for latent grid {grid_key}")
    grid_reference = references[grid_key]
    effective_levels = anchor_loss._effective_levels(x0_pred.shape[-2], x0_pred.shape[-1], levels)
    pyramid = anchor_loss._laplacian_pyramid(x0_pred, effective_levels)
    if len(pyramid) != len(grid_reference["bands"]):
        raise ValueError(f"adv reference band count mismatch for grid {grid_key}")

    losses = []
    for band, band_reference in zip(pyramid, grid_reference["bands"]):
        tokens = tokenize_band(band, tokens_per_band)
        embeddings = _rff_features(tokens, band_reference).mean(dim=1)
        mu = band_reference["mu"].to(device=embeddings.device, dtype=torch.float32)
        losses.append((embeddings - mu.view(1, -1)).square().sum(dim=-1))
    return torch.stack(losses, dim=0).mean(dim=0)


def adv_snr_weights(
    timesteps: torch.Tensor,
    mode: str,
    noise_scheduler=None,
    timesteps_in_sigma: bool = False,
) -> torch.Tensor:
    """Batch-mean-one SNR weighting for the adv term, detached from autograd."""
    snr = hf_snr_from_timesteps(
        timesteps,
        mode,
        noise_scheduler=noise_scheduler,
        timesteps_in_sigma=timesteps_in_sigma,
    ).float()
    weights = snr / snr.mean().clamp_min(ADV_EPS)
    return weights.detach()
