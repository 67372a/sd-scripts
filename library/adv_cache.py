"""Build, validate, and load stationary RFF-MMD reference caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import torch

from library import anchor_loss, dist_match

CACHE_VERSION = 1


def fingerprint_payload(payload: Mapping) -> str:
    """Stable SHA-256 digest for JSON-compatible dataset/cache metadata."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        return torch.load(path, map_location="cpu")


def _validate_loaded_cache(cache, signature: str) -> bool:
    if not isinstance(cache, dict) or cache.get("version") != CACHE_VERSION:
        return False
    if cache.get("signature") != signature:
        return False
    references = cache.get("references")
    if not isinstance(references, dict) or not references:
        return False
    for grid_key, grid in references.items():
        if not isinstance(grid_key, str) or not isinstance(grid, dict) or not isinstance(grid.get("bands"), list):
            return False
        if not grid["bands"]:
            return False
        for band in grid["bands"]:
            if not isinstance(band, dict) or not all(k in band for k in ("mean", "std", "sigmas", "omega", "phase", "mu")):
                return False
            tensors = [band[k] for k in ("mean", "std", "sigmas", "omega", "phase", "mu")]
            if any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in tensors):
                return False
            if band["mean"].ndim != 1 or band["std"].shape != band["mean"].shape:
                return False
            if band["omega"].ndim != 3 or band["omega"].shape[0] != band["sigmas"].numel():
                return False
            if band["phase"].shape != band["omega"].shape[:2]:
                return False
            if band["mu"].numel() != band["omega"].shape[0] * band["omega"].shape[1]:
                return False
    return True


def load_adv_cache(path: str, signature: str, device) -> Dict[str, Dict]:
    """Load and validate a cache, raising ``ValueError`` for stale/corrupt files."""
    try:
        cache = _safe_torch_load(path)
    except Exception as exc:
        raise ValueError(f"could not read stationary adv cache: {path}") from exc
    if not _validate_loaded_cache(cache, signature):
        raise ValueError(f"invalid or stale stationary adv cache: {path}")
    return {
        grid_key: {
            "bands": [
                {key: value.to(device=device) for key, value in band.items()}
                for band in grid["bands"]
            ]
        }
        for grid_key, grid in cache["references"].items()
    }


def _as_chw(latent: torch.Tensor, device) -> torch.Tensor:
    if not isinstance(latent, torch.Tensor):
        latent = torch.as_tensor(latent)
    if latent.ndim == 5 and latent.shape[0] == 1 and latent.shape[2] == 1:
        latent = latent[0, :, 0]
    if latent.ndim == 4:
        if latent.shape[0] == 1:
            latent = latent[0]
        elif latent.shape[1] == 1:  # individual Anima latent [C,1,H,W]
            latent = latent[:, 0]
        else:
            raise ValueError("reference cache source must be an individual CHW latent")
    if latent.ndim != 3:
        raise ValueError(f"expected CHW latent, got shape {tuple(latent.shape)}")
    return latent.detach().to(device=device)


def _create_band_reference(
    sources: Sequence[Callable[[], torch.Tensor]],
    height: int,
    width: int,
    band_index: int,
    levels: int,
    token_budget: int,
    reference_samples: int,
    features: int,
    explicit_sigmas: List[float] | None,
    seed: int,
    device,
) -> Dict[str, torch.Tensor]:
    effective_levels = anchor_loss._effective_levels(height, width, levels)
    channel_sum = channel_sq_sum = None
    token_count = 0
    samples = []
    # Each band has an independent, deterministic sampling stream on the active
    # accelerator. This generator never touches the training RNG state.
    sample_generator = torch.Generator(device=device)
    sample_generator.manual_seed(int(seed) + 0x1009 * (band_index + 1))
    total_source_weight = sum(max(1, int(getattr(source, "adv_weight", 1))) for source in sources)

    # Pass one: reference whitening moments and a bounded token sample for the
    # median bandwidth heuristic.
    with torch.no_grad():
        for source in sources:
            source_weight = max(1, int(getattr(source, "adv_weight", 1)))
            clean = _as_chw(source(), device).unsqueeze(0)
            if clean.shape[-2:] != (height, width):
                raise ValueError(
                    f"adv source grid hint was inconsistent: expected {height}x{width}, "
                    f"got {clean.shape[-2]}x{clean.shape[-1]}"
                )
            band = anchor_loss._laplacian_pyramid(clean, effective_levels)[band_index]
            tokens = dist_match.tokenize_band(band, token_budget)[0].float()
            current_sum = tokens.sum(dim=0)
            current_sq_sum = tokens.square().sum(dim=0)
            channel_sum = current_sum * source_weight if channel_sum is None else channel_sum + current_sum * source_weight
            channel_sq_sum = current_sq_sum * source_weight if channel_sq_sum is None else channel_sq_sum + current_sq_sum * source_weight
            token_count += tokens.shape[0] * source_weight
            per_source_sample = max(1, math.ceil(reference_samples * source_weight / max(total_source_weight, 1)))
            sample_count = min(tokens.shape[0], per_source_sample)
            indices = torch.randperm(tokens.shape[0], generator=sample_generator, device=device)[:sample_count]
            samples.append(tokens.index_select(0, indices))

    mean = channel_sum / max(token_count, 1)
    variance = (channel_sq_sum / max(token_count, 1) - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(dist_match.ADV_EPS)
    reference_tokens = torch.cat(samples, dim=0)
    if reference_tokens.shape[0] > reference_samples:
        take = torch.randperm(reference_tokens.shape[0], generator=sample_generator, device=device)[:reference_samples]
        reference_tokens = reference_tokens.index_select(0, take)
    reference_tokens = (reference_tokens - mean) / std

    if explicit_sigmas is None:
        if reference_tokens.shape[0] > 1:
            pair_distances = torch.pdist(reference_tokens.float(), p=2)
            median = pair_distances.median().clamp_min(dist_match.ADV_EPS)
        else:
            median = torch.ones((), device=device, dtype=torch.float32)
        sigmas = torch.stack([median * multiplier for multiplier in dist_match.DEFAULT_SIGMA_MULTIPLIERS])
    else:
        sigmas = torch.tensor(explicit_sigmas, device=device, dtype=torch.float32)

    # Frequencies and phases are fixed cache contents, generated independently
    # for each band and never sampled by the training step.
    rff_generator = torch.Generator(device=device)
    rff_generator.manual_seed(int(seed) + 0x51ED * (band_index + 1))
    omega = torch.randn(
        (sigmas.numel(), features, mean.numel()), generator=rff_generator, device=device, dtype=torch.float32
    ) / sigmas.view(-1, 1, 1)
    phase = torch.rand(
        (sigmas.numel(), features), generator=rff_generator, device=device, dtype=torch.float32
    ) * (2.0 * math.pi)

    # Pass two: accumulate the fixed reference kernel-mean embedding. Each
    # source contributes all pooled tokens, with no reference token set retained.
    embedding_sum = torch.zeros(sigmas.numel() * features, device=device, dtype=torch.float32)
    embedding_count = 0
    band_reference = {"mean": mean, "std": std, "omega": omega, "phase": phase}
    with torch.no_grad():
        for source in sources:
            source_weight = max(1, int(getattr(source, "adv_weight", 1)))
            clean = _as_chw(source(), device).unsqueeze(0)
            if clean.shape[-2:] != (height, width):
                raise ValueError(
                    f"adv source grid hint was inconsistent: expected {height}x{width}, "
                    f"got {clean.shape[-2]}x{clean.shape[-1]}"
                )
            band = anchor_loss._laplacian_pyramid(clean, effective_levels)[band_index]
            tokens = dist_match.tokenize_band(band, token_budget)
            embeddings = dist_match._rff_features(tokens, band_reference)[0]
            embedding_sum.add_(embeddings.sum(dim=0) * source_weight)
            embedding_count += embeddings.shape[0] * source_weight
    mu = embedding_sum / max(embedding_count, 1)
    return {
        "mean": mean.detach(),
        "std": std.detach(),
        "sigmas": sigmas.detach(),
        "omega": omega.detach(),
        "phase": phase.detach(),
        "mu": mu.detach(),
    }


def build_adv_references(
    sources: Sequence[Callable[[], torch.Tensor]],
    levels: int,
    token_budget: int,
    reference_samples: int,
    features: int,
    bandwidths: str,
    seed: int,
    device,
) -> Dict[str, Dict]:
    """Build a reference per observed clean-input grid using two streaming passes."""
    if not sources:
        raise ValueError("cannot build adv reference: the training dataset has no clean latent samples")
    explicit_sigmas = dist_match.parse_bandwidths(bandwidths)
    grouped: Dict[str, List[Callable[[], torch.Tensor]]] = {}
    # Determine the grid from the first load of each source, then replay it for
    # the two streaming statistics/embedding passes. Dataset latents themselves
    # are not retained by this builder.
    for source in sources:
        grid_key = getattr(source, "adv_grid_key", None)
        if grid_key is None:
            latent = _as_chw(source(), device)
            grid_key = f"{latent.shape[-2]}x{latent.shape[-1]}"
        grouped.setdefault(grid_key, []).append(source)

    references = {}
    for grid_key, grid_sources in grouped.items():
        height, width = map(int, grid_key.split("x"))
        effective_levels = anchor_loss._effective_levels(height, width, levels)
        refs = []
        for band_index in range(effective_levels + 1):
            refs.append(
                _create_band_reference(
                    grid_sources,
                    height,
                    width,
                    band_index,
                    levels,
                    token_budget,
                    reference_samples,
                    features,
                    explicit_sigmas,
                    seed,
                    device,
                )
            )
        references[grid_key] = {"bands": refs}
    return references


def save_adv_cache(path: str, signature: str, references: Dict[str, Dict]) -> None:
    """Atomically write a CPU-only cache so it can be shared by future runs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cpu_references = {
        grid_key: {
            "bands": [{key: value.detach().float().cpu() for key, value in band.items()} for band in grid["bands"]]
        }
        for grid_key, grid in references.items()
    }
    cache = {"version": CACHE_VERSION, "signature": signature, "references": cpu_references}
    fd, temporary_path = tempfile.mkstemp(prefix=".adv-cache-", suffix=".pt", dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        torch.save(cache, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
