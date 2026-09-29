"""CUDA coverage for stationary multi-scale random-feature MMD matching."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from library import adv_cache, dist_match


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="stationary adv tests require CUDA")


def _randn(shape, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(shape, device="cuda", generator=generator)


def _build_reference(samples=None):
    if samples is None:
        samples = [_randn((4, 16, 12), seed) for seed in (101, 102, 103)]
    sources = [lambda value=value: value for value in samples]
    return adv_cache.build_adv_references(
        sources,
        levels=2,
        token_budget=24,
        reference_samples=32,
        features=16,
        bandwidths="",
        seed=0x5A17C0DE,
        device=torch.device("cuda"),
    )


def test_pool_tokenizer_is_bounded_and_differentiable():
    image = _randn((2, 4, 31, 19), 1).requires_grad_(True)
    tokens = dist_match.tokenize_band(image, token_budget=64)
    assert tokens.shape[0] == 2
    assert tokens.shape[1] <= 64
    assert tokens.shape[2] == 4
    tokens.square().mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_reference_build_loss_gradient_and_rng_neutrality():
    samples = [_randn((4, 16, 12), seed) for seed in (11, 12, 13)]
    rng_state = torch.cuda.get_rng_state()
    references = _build_reference(samples)
    assert torch.equal(rng_state, torch.cuda.get_rng_state())
    assert set(references) == {"16x12"}
    assert len(references["16x12"]["bands"]) == 3
    for band in references["16x12"]["bands"]:
        assert band["mu"].device.type == "cuda"
        assert band["mu"].numel() == band["omega"].shape[0] * band["omega"].shape[1]
        assert torch.isfinite(band["mu"]).all()

    predicted = samples[0].unsqueeze(0).detach().requires_grad_(True)
    per_sample = dist_match.adv_per_sample_loss(predicted, references, levels=2, tokens_per_band=24)
    assert per_sample.shape == (1,)
    assert torch.isfinite(per_sample).all()
    assert per_sample.min() >= 0.0
    assert per_sample.max() <= 2.0001
    per_sample.mean().backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.abs().sum() > 0


def test_trainer_adv_term_decomposes_loss_and_backpropagates():
    train_network = pytest.importorskip("train_network")
    references = _build_reference()
    trainer = train_network.NetworkTrainer()
    trainer.adv_scale = 0.35
    trainer.adv_levels = 2
    trainer.adv_tokens_per_band = 24
    trainer.adv_prediction_mode = "x0_direct"
    trainer.adv_references = references
    trainer.post_process_loss = lambda loss, *args: loss

    predicted_clean = _randn((1, 4, 16, 12), 87).requires_grad_(True)
    initial_loss = torch.tensor(1.75, device="cuda", requires_grad=True)
    expected_adv = dist_match.adv_per_sample_loss(
        predicted_clean, references, levels=2, tokens_per_band=24
    ).mean() * trainer.adv_scale
    final_loss = trainer._apply_adv_term(
        initial_loss,
        predicted_clean,
        clean=predicted_clean.detach(),
        timesteps=torch.tensor([500], device="cuda"),
        weighting=None,
        noise_scheduler=None,
        args=SimpleNamespace(),
        batch={"loss_weights": torch.ones(1, device="cuda")},
    )
    assert torch.allclose(final_loss, initial_loss + expected_adv)
    assert torch.allclose(trainer.adv_loss_value, expected_adv.detach())
    final_loss.backward()
    assert predicted_clean.grad is not None
    assert torch.isfinite(predicted_clean.grad).all()


def test_cache_round_trip_and_signature_validation(tmp_path):
    references = _build_reference()
    path = tmp_path / "stationary-adv.pt"
    adv_cache.save_adv_cache(str(path), "test-signature", references)
    loaded = adv_cache.load_adv_cache(str(path), "test-signature", torch.device("cuda"))
    assert loaded.keys() == references.keys()
    for original_band, loaded_band in zip(references["16x12"]["bands"], loaded["16x12"]["bands"]):
        assert loaded_band["mu"].device.type == "cuda"
        assert torch.allclose(original_band["mu"], loaded_band["mu"])
    with pytest.raises(ValueError, match="invalid or stale"):
        adv_cache.load_adv_cache(str(path), "different-signature", torch.device("cuda"))


def test_trainer_builds_then_reuses_dataset_reference_cache(tmp_path):
    train_network = pytest.importorskip("train_network")
    latent = _randn((4, 16, 12), 55)
    info = SimpleNamespace(
        is_val=False,
        is_reg=False,
        latents=latent,
        latents_flipped=None,
        latents_aug_variants=None,
        latents_by_reso={},
        latents_npz=None,
        absolute_path="synthetic-training-sample.png",
        bucket_reso=(96, 128),
        jitter_bucket_info={},
    )
    subset = SimpleNamespace(
        is_val=False,
        is_reg=False,
        flip_aug=False,
        color_aug=False,
        gamma_aug=False,
        random_crop=False,
        num_repeats=1,
    )
    dataset = SimpleNamespace(
        image_data={"sample": info},
        image_to_subset={"sample": subset},
        latents_caching_strategy=None,
    )
    dataset_group = SimpleNamespace(datasets=[dataset])
    accelerator = SimpleNamespace(
        is_main_process=True,
        device=torch.device("cuda"),
        wait_for_everyone=lambda: None,
    )
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        adv_cache_dir=None,
        pretrained_model_name_or_path="synthetic-base",
        vae=None,
        vae_custom_scale=None,
        vae_custom_shift=None,
    )
    trainer = train_network.NetworkTrainer()
    trainer.adv_scale = 0.25
    trainer.adv_levels = 2
    trainer.adv_features = 16
    trainer.adv_tokens_per_band = 24
    trainer.adv_reference_samples = 32
    trainer.adv_bandwidths = ""
    trainer.adv_rff_seed = 19

    trainer.prepare_adv_reference(args, dataset_group, accelerator)
    first_mu = trainer.adv_references["16x12"]["bands"][0]["mu"].clone()
    trainer.adv_references = None
    trainer.prepare_adv_reference(args, dataset_group, accelerator)
    assert trainer.adv_references is not None
    assert torch.allclose(first_mu, trainer.adv_references["16x12"]["bands"][0]["mu"])

    cache_file = next((tmp_path / "adv_cache").glob("reference-*.pt"))
    cache_file.write_bytes(b"corrupt cache")
    trainer.adv_references = None
    trainer.prepare_adv_reference(args, dataset_group, accelerator)
    assert torch.allclose(first_mu, trainer.adv_references["16x12"]["bands"][0]["mu"])


def test_bandwidth_parser_and_validation():
    assert dist_match.parse_bandwidths("") is None
    assert dist_match.parse_bandwidths("0.5, 1,2") == [0.5, 1.0, 2.0]
    with pytest.raises(ValueError, match="positive"):
        dist_match.parse_bandwidths("1,0")
    with pytest.raises(ValueError, match="adv_levels"):
        dist_match.validate_adv_args(0.1, 0, 16, 24, 32)


def test_snr_gate_is_detached_batch_mean_one():
    timesteps = torch.tensor([0.1, 0.5, 0.9], device="cuda")
    weights = dist_match.adv_snr_weights(timesteps, mode="flow", timesteps_in_sigma=True)
    assert weights.device.type == "cuda"
    assert not weights.requires_grad
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-6)


def test_cli_defaults_and_explicit_values():
    train_network = pytest.importorskip("train_network")
    parser = train_network.setup_parser()
    defaults = parser.parse_args([])
    assert defaults.adv_scale == 0.0
    assert defaults.adv_levels == 4
    assert defaults.adv_domain == "latent"
    assert defaults.adv_features == 256
    assert defaults.adv_bandwidths == ""
    assert defaults.adv_tokens_per_band == 1024
    assert defaults.adv_reference_samples == 4096
    assert defaults.adv_rff_seed == 0x5A17C0DE
    assert defaults.adv_snr_weighting is False

    explicit = parser.parse_args(
        [
            "--adv_scale", "0.25", "--adv_levels", "3", "--adv_domain", "latent",
            "--adv_features", "64", "--adv_bandwidths", "0.25,0.5,1.0",
            "--adv_tokens_per_band", "128", "--adv_reference_samples", "512",
            "--adv_rff_seed", "0x1234", "--adv_snr_weighting",
        ]
    )
    assert explicit.adv_scale == 0.25
    assert explicit.adv_levels == 3
    assert explicit.adv_features == 64
    assert explicit.adv_rff_seed == 0x1234
    assert explicit.adv_snr_weighting is True


def test_objective_validation_and_prediction_mode_resolution():
    train_network = pytest.importorskip("train_network")
    trainer = train_network.NetworkTrainer()
    trainer.hf_prediction_mode = "flow"
    trainer.hf_timesteps_in_sigma = True
    trainer.setup_adv_objective(
        argparse.Namespace(
            adv_scale=0.5,
            adv_levels=3,
            adv_domain="latent",
            adv_features=32,
            adv_bandwidths="",
            adv_tokens_per_band=64,
            adv_reference_samples=128,
            adv_rff_seed=9,
            adv_snr_weighting=True,
            flow_model=False,
            v_parameterization=False,
        )
    )
    assert trainer.adv_prediction_mode == "flow"
    assert trainer.adv_timesteps_in_sigma is True
    assert trainer.adv_scale == 0.5

    with pytest.raises(ValueError, match="adv_scale"):
        trainer.setup_adv_objective(
            argparse.Namespace(
                adv_scale=-1.0,
                adv_levels=3,
                adv_domain="latent",
                adv_features=32,
                adv_bandwidths="",
                adv_tokens_per_band=64,
                adv_reference_samples=128,
                adv_rff_seed=9,
                adv_snr_weighting=False,
                flow_model=False,
                v_parameterization=False,
            )
        )
