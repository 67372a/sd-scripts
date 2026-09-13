import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from library.config_util import (
    BlueprintGenerator,
    ConfigSanitizer,
    DreamBoothDatasetParams,
    DreamBoothSubsetParams,
)
from library.train_util import BaseDataset, DreamBoothDataset, DreamBoothSubset, ImageInfo


def _subset(resolution=None, batch_size=None, flip_aug=False):
    return SimpleNamespace(
        resolution=resolution,
        batch_size=batch_size,
        flip_aug=flip_aug,
    )


def _dataset(resolution=(512, 512), batch_size=1, enable_bucket=False):
    dataset = BaseDataset(resolution, 1.0, False, False)
    dataset.batch_size = batch_size
    dataset.enable_bucket = enable_bucket
    dataset.min_bucket_reso = 256 if enable_bucket else None
    dataset.max_bucket_reso = 1024 if enable_bucket else None
    dataset.bucket_reso_steps = 64 if enable_bucket else None
    dataset.bucket_no_upscale = False
    dataset.multires_training = False
    return dataset


def _add_image(dataset, key, subset, image_size, num_repeats=1):
    info = ImageInfo(key, num_repeats, "caption", False, False, key)
    info.image_size = image_size
    dataset.register_image(info, subset)
    return info


def _batch_slices(dataset):
    """Emulate the index arithmetic of BaseDataset.__getitem__ without running the full pipeline."""
    slices = []
    for index in range(len(dataset)):
        batch_index = dataset.buckets_indices[index]
        bucket = dataset.batch_buckets[batch_index.bucket_index]
        image_index = batch_index.batch_index * batch_index.bucket_batch_size
        slices.append(bucket[image_index : image_index + batch_index.bucket_batch_size])
    return slices


def test_subset_schema_accepts_batch_size():
    sanitizer = ConfigSanitizer(True, False, False, True)
    config = {
        "general": {"resolution": 512},
        "datasets": [
            {
                "batch_size": 4,
                "subsets": [
                    {
                        "image_dir": "images",
                        "batch_size": 2,
                    }
                ],
            }
        ],
    }

    sanitized = sanitizer.sanitize_user_config(config)
    subset = sanitized["datasets"][0]["subsets"][0]
    assert subset["batch_size"] == 2


def test_subset_batch_size_params_override_dataset_and_fallback_when_undefined():
    explicit = BlueprintGenerator.generate_params_by_fallbacks(
        DreamBoothSubsetParams,
        [{"image_dir": "images", "batch_size": 2}, {"batch_size": 4}],
    )
    inherited = BlueprintGenerator.generate_params_by_fallbacks(
        DreamBoothSubsetParams,
        [{"image_dir": "images"}, {"batch_size": 4}],
    )

    assert explicit.batch_size == 2
    assert inherited.batch_size == 4


def test_get_subset_batch_size_fallback_and_override():
    dataset = _dataset(batch_size=4)
    overridden = _subset(None, 2)
    inherited = _subset(None, None)

    assert dataset.get_subset_batch_size(overridden) == 2
    assert dataset.get_subset_batch_size(inherited) == 4


def test_same_resolution_subsets_are_batched_separately_with_own_batch_sizes():
    dataset = _dataset((512, 512), batch_size=4)
    subset_a = _subset(None, 2)
    subset_b = _subset(None, 3)
    for i in range(5):
        _add_image(dataset, f"a{i}", subset_a, (512, 512))
    for i in range(5):
        _add_image(dataset, f"b{i}", subset_b, (512, 512))
    dataset.subsets = [subset_a, subset_b]

    dataset.make_buckets()

    # ceil(5/2) + ceil(5/3) = 3 + 2 batches
    assert len(dataset) == 5

    keys_a = {f"a{i}" for i in range(5)}
    keys_b = {f"b{i}" for i in range(5)}
    slices = _batch_slices(dataset)
    for batch in slices:
        keys = set(batch)
        assert keys <= keys_a or keys <= keys_b, "batch mixes images from multiple subsets"
        if keys <= keys_a:
            assert len(batch) <= 2
        else:
            assert len(batch) <= 3

    # all images are still consumed exactly once
    all_keys = [k for batch in slices for k in batch]
    assert len(all_keys) == 10
    assert set(all_keys) == keys_a | keys_b


def test_batch_size_falls_back_to_dataset_level_for_undefined_subsets():
    dataset = _dataset((512, 512), batch_size=3)
    subset_a = _subset(None, None)
    subset_b = _subset(None, None)
    for i in range(4):
        _add_image(dataset, f"a{i}", subset_a, (512, 512))
    for i in range(4):
        _add_image(dataset, f"b{i}", subset_b, (512, 512))
    dataset.subsets = [subset_a, subset_b]

    dataset.make_buckets()

    # both subsets inherit the dataset batch size, so their configurations match and
    # they share a pool: ceil(8/3) = 3 batches (instead of 2 + 2 with separate pools)
    assert len(dataset) == 3
    for batch in _batch_slices(dataset):
        assert len(batch) <= 3


def test_same_config_subsets_share_batch_pool():
    dataset = _dataset((512, 512), batch_size=4)
    subset_a = _subset(None, None)
    subset_b = _subset(None, None)
    for i in range(5):
        _add_image(dataset, f"a{i}", subset_a, (512, 512))
    for i in range(5):
        _add_image(dataset, f"b{i}", subset_b, (512, 512))
    dataset.subsets = [subset_a, subset_b]

    dataset.make_buckets()

    # matching configurations merge into one pool: ceil(10/4) = 3 batches
    # (subset-scoped pools would need 2 + 2 = 4 batches)
    assert len(dataset) == 3

    keys_a = {f"a{i}" for i in range(5)}
    keys_b = {f"b{i}" for i in range(5)}
    batches = _batch_slices(dataset)
    for batch in batches:
        assert len(batch) <= 4

    # all images are consumed exactly once
    all_keys = [k for batch in batches for k in batch]
    assert len(all_keys) == 10
    assert set(all_keys) == keys_a | keys_b

    # the representative subset of the shared pool reports the effective batch size
    assert len(dataset.batch_buckets) == 1
    assert dataset.get_subset_batch_size(dataset.batch_bucket_subsets[0]) == 4


def test_explicitly_equal_batch_sizes_share_batch_pool():
    dataset = _dataset((512, 512), batch_size=1)
    subset_a = _subset(None, 2)
    subset_b = _subset(None, 2)
    for i in range(3):
        _add_image(dataset, f"a{i}", subset_a, (512, 512))
        _add_image(dataset, f"b{i}", subset_b, (512, 512))
    dataset.subsets = [subset_a, subset_b]

    dataset.make_buckets()

    # equal effective batch sizes merge: 6 items / 2 = 3 batches
    assert len(dataset) == 3
    assert len(dataset.batch_buckets) == 1
    for batch in _batch_slices(dataset):
        assert len(batch) <= 2


def test_differing_flip_aug_keeps_pools_separate():
    dataset = _dataset((512, 512), batch_size=2)
    subset_a = _subset(None, 2, flip_aug=False)
    subset_b = _subset(None, 2, flip_aug=True)
    for i in range(4):
        _add_image(dataset, f"a{i}", subset_a, (512, 512))
        _add_image(dataset, f"b{i}", subset_b, (512, 512))
    dataset.subsets = [subset_a, subset_b]

    dataset.make_buckets()

    # flip_aug differs, so the aug-flag assert in __getitem__ would fail on mixed
    # batches: the subsets must keep separate pools despite equal batch sizes
    assert len(dataset.batch_buckets) == 2
    assert len(dataset) == 4

    keys_a = {f"a{i}" for i in range(4)}
    keys_b = {f"b{i}" for i in range(4)}
    for batch in _batch_slices(dataset):
        keys = set(batch)
        assert keys <= keys_a or keys <= keys_b, "batch mixes images with differing flip_aug"


def test_differing_resolution_keeps_pools_separate():
    dataset = _dataset((512, 512), batch_size=2)
    subset_a = _subset((512, 512), 2)
    subset_b = _subset((768, 768), 2)
    for i in range(4):
        _add_image(dataset, f"a{i}", subset_a, (512, 512))
        _add_image(dataset, f"b{i}", subset_b, (768, 768))
    dataset.subsets = [subset_a, subset_b]

    dataset.make_buckets()

    # resolutions differ, so the subsets keep separate pools despite equal batch sizes
    assert len(dataset.batch_buckets) == 2
    assert len(dataset) == 4

    keys_a = {f"a{i}" for i in range(4)}
    keys_b = {f"b{i}" for i in range(4)}
    for batch in _batch_slices(dataset):
        keys = set(batch)
        assert keys <= keys_a or keys <= keys_b, "batch mixes images with differing resolutions"
        assert dataset.image_data[next(iter(keys))].bucket_reso in {(512, 512), (768, 768)}


def test_num_repeats_are_preserved_in_subset_scoped_buckets():
    dataset = _dataset((512, 512), batch_size=2)
    subset_a = _subset(None, 2)
    _add_image(dataset, "a0", subset_a, (512, 512), num_repeats=3)
    _add_image(dataset, "a1", subset_a, (512, 512), num_repeats=1)
    dataset.subsets = [subset_a]

    dataset.make_buckets()

    # 3 + 1 = 4 items with batch size 2 -> 2 batches
    assert len(dataset) == 2
    all_keys = [k for batch in _batch_slices(dataset) for k in batch]
    assert len(all_keys) == 4
    assert all_keys.count("a0") == 3
    assert all_keys.count("a1") == 1


def test_epoch_shuffle_keeps_batches_subset_scoped():
    dataset = _dataset((512, 512), batch_size=4)
    subset_a = _subset(None, 2)
    subset_b = _subset(None, 3)
    for i in range(5):
        _add_image(dataset, f"a{i}", subset_a, (512, 512))
    for i in range(5):
        _add_image(dataset, f"b{i}", subset_b, (512, 512))
    dataset.subsets = [subset_a, subset_b]

    dataset.make_buckets()
    dataset.current_epoch = 1
    dataset.shuffle_buckets()

    keys_a = {f"a{i}" for i in range(5)}
    keys_b = {f"b{i}" for i in range(5)}
    for batch in _batch_slices(dataset):
        keys = set(batch)
        assert keys <= keys_a or keys <= keys_b, "shuffled batch mixes images from multiple subsets"


def test_batch_formation_is_deterministic_across_constructions():
    def build():
        dataset = _dataset((512, 512), batch_size=4)
        subset_a = _subset(None, 2)
        subset_b = _subset(None, 3)
        for i in range(5):
            _add_image(dataset, f"a{i}", subset_a, (512, 512))
        for i in range(5):
            _add_image(dataset, f"b{i}", subset_b, (512, 512))
        dataset.subsets = [subset_a, subset_b]
        dataset.make_buckets()
        return dataset

    first, second = build(), build()

    assert first.batch_buckets == second.batch_buckets
    assert [(b.bucket_index, b.bucket_batch_size, b.batch_index) for b in first.buckets_indices] == [
        (b.bucket_index, b.bucket_batch_size, b.batch_index) for b in second.buckets_indices
    ]
    # each batch bucket is owned by a single subset and matches its batch size
    for bucket_index, bucket in enumerate(first.batch_buckets):
        subset = first.batch_bucket_subsets[bucket_index]
        assert first.get_subset_batch_size(subset) in (2, 3)
        assert all(first.image_to_subset[k] is subset for k in bucket)


def test_batch_iteration_does_not_consume_global_rng():
    # validation datasets rely on batch formation being pure index arithmetic:
    # iterating all batches must not advance the global random state
    dataset = _dataset((512, 512), batch_size=4)
    subset_a = _subset(None, 2)
    subset_b = _subset(None, 3)
    for i in range(5):
        _add_image(dataset, f"a{i}", subset_a, (512, 512))
    for i in range(5):
        _add_image(dataset, f"b{i}", subset_b, (512, 512))
    dataset.subsets = [subset_a, subset_b]

    dataset.make_buckets()

    state = random.getstate()
    _batch_slices(dataset)  # emulate the __getitem__ slicing
    assert random.getstate() == state


def test_dreambooth_dataset_constructs_with_subset_batch_size(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    for i in range(3):
        Image.new("RGB", (512, 512)).save(dir_a / f"a{i}.png")
        Image.new("RGB", (512, 512)).save(dir_b / f"b{i}.png")

    params_a = DreamBoothSubsetParams(image_dir=str(dir_a), batch_size=2)
    params_b = DreamBoothSubsetParams(image_dir=str(dir_b))
    dataset_params = DreamBoothDatasetParams(
        resolution=(512, 512),
        enable_bucket=False,
        batch_size=4,
    )
    dataset = DreamBoothDataset(
        [DreamBoothSubset(**asdict(params_a)), DreamBoothSubset(**asdict(params_b))],
        is_training_dataset=True,
        **asdict(dataset_params),
    )

    dataset.make_buckets()

    # subset a: ceil(3/2) = 2 batches, subset b: ceil(3/4) = 1 batch
    assert len(dataset) == 3

    for batch in _batch_slices(dataset):
        parents = {Path(k).parent.name for k in batch}
        assert len(parents) == 1, "batch mixes images from multiple subsets"
        if parents == {"a"}:
            assert len(batch) <= 2
        else:
            assert len(batch) <= 4
