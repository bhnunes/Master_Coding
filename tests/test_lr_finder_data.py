from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from helpers.lr_finder import data as lr_data
from helpers.lr_finder.config import BCEDiceSearchSpace, LRFinderConfig, ModelPlan


def _build_config(tmp_path: Path, *, smart_sampling: bool = False) -> LRFinderConfig:
    return LRFinderConfig(
        hdf5_drive_dir=tmp_path / "data",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "local",
        stage_input_locally=False,
        overwrite_output=True,
        smart_sampling=smart_sampling,
        execution_mode="PAPER",
        amp_precision="fp32",
        seed=24,
        batch_size=2,
        workers=0,
        use_subset=False,
        subset_ratio=1.0,
        num_lhs_samples=2,
        end_lr=0.1,
        num_iter=5,
        num_repeats=1,
        optimizer_weight_decay=1e-4,
        optimizer_start_lr=1e-8,
        pdf_name="report.pdf",
        hf_token=None,
        search_space=BCEDiceSearchSpace(),
        model_plans=[ModelPlan(architecture="FPN", encoder="resnet34")],
    )


def _write_split_shards(
    root: Path, split_name: str, patient_ids: list[str]
) -> lr_data.ShardDatasetLayout:
    shard_dir = root / split_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for index, patient_id in enumerate(patient_ids):
        relative_hdf5_path = f"{split_name}/{patient_id}.h5"
        with h5py.File(shard_dir / f"{patient_id}.h5", "w") as handle:
            handle.create_dataset(
                "images",
                data=np.full((1, 4, 4, 3), index + 1, dtype=np.uint8),
            )
            handle.create_dataset(
                "masks",
                data=np.full((1, 4, 4), index % 2, dtype=np.uint8),
            )
            handle.create_dataset("labels", data=np.array([index % 2], dtype=np.uint8))
            handle.create_dataset("patient_ids", data=np.array([patient_id.encode()]))
            handle.create_dataset("filenames", data=np.array([f"{patient_id}.png".encode()]))
            handle.attrs["source_hdf5_sha256"] = f"{split_name}-sha"
            handle.attrs["upstream_source_signature"] = f"{split_name}-sig"
        manifest_rows.append(
            {
                "split": split_name.replace("_shards", ""),
                "patient_id": patient_id,
                "relative_hdf5_path": relative_hdf5_path,
                "rows": 1,
                "label_0_count": 1 if index % 2 == 0 else 0,
                "label_1_count": 1 if index % 2 == 1 else 0,
            }
        )
        sample_rows.append(
            {
                "split": split_name.replace("_shards", ""),
                "patient_id": patient_id,
                "relative_hdf5_path": relative_hdf5_path,
                "row_in_shard": 0,
                "label": index % 2,
                "filename": f"{patient_id}.png",
            }
        )
    pq.write_table(pa.Table.from_pylist(manifest_rows), shard_dir / "manifest.parquet")
    pq.write_table(pa.Table.from_pylist(sample_rows), shard_dir / "sample_manifest.parquet")
    return lr_data.ShardDatasetLayout(
        shard_dir=shard_dir,
        manifest_path=shard_dir / "manifest.parquet",
        sample_manifest_path=shard_dir / "sample_manifest.parquet",
        local_cache_dir=None,
    )


def _write_multirow_train_shards(root: Path) -> lr_data.ShardDatasetLayout:
    shard_dir = root / "TRAIN_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(shard_dir / "p1.h5", "w") as handle:
        handle.create_dataset(
            "images",
            data=np.stack(
                [
                    np.full((4, 4, 3), 11, dtype=np.uint8),
                    np.full((4, 4, 3), 22, dtype=np.uint8),
                ]
            ),
        )
        handle.create_dataset(
            "masks",
            data=np.stack(
                [
                    np.zeros((4, 4), dtype=np.uint8),
                    np.ones((4, 4), dtype=np.uint8),
                ]
            ),
        )
        handle.create_dataset("labels", data=np.array([0, 1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([b"p1", b"p1"], dtype="S16"))
        handle.create_dataset("filenames", data=np.array([b"p1-a.png", b"p1-b.png"], dtype="S32"))
        handle.attrs["source_hdf5_sha256"] = "train-sha"
        handle.attrs["upstream_source_signature"] = "train-sig"

    with h5py.File(shard_dir / "p2.h5", "w") as handle:
        handle.create_dataset("images", data=np.full((1, 4, 4, 3), 33, dtype=np.uint8))
        handle.create_dataset("masks", data=np.full((1, 4, 4), 2, dtype=np.uint8))
        handle.create_dataset("labels", data=np.array([1], dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([b"p2"], dtype="S16"))
        handle.create_dataset("filenames", data=np.array([b"p2-a.png"], dtype="S32"))
        handle.attrs["source_hdf5_sha256"] = "train-sha"
        handle.attrs["upstream_source_signature"] = "train-sig"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "split": "TRAIN",
                    "patient_id": "p1",
                    "relative_hdf5_path": "TRAIN_shards/p1.h5",
                    "rows": 2,
                    "label_0_count": 1,
                    "label_1_count": 1,
                },
                {
                    "split": "TRAIN",
                    "patient_id": "p2",
                    "relative_hdf5_path": "TRAIN_shards/p2.h5",
                    "rows": 1,
                    "label_0_count": 0,
                    "label_1_count": 1,
                },
            ]
        ),
        shard_dir / "manifest.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "split": "TRAIN",
                    "patient_id": "p1",
                    "relative_hdf5_path": "TRAIN_shards/p1.h5",
                    "row_in_shard": 1,
                    "label": 1,
                    "filename": "p1-b.png",
                },
                {
                    "split": "TRAIN",
                    "patient_id": "p1",
                    "relative_hdf5_path": "TRAIN_shards/p1.h5",
                    "row_in_shard": 0,
                    "label": 0,
                    "filename": "p1-a.png",
                },
                {
                    "split": "TRAIN",
                    "patient_id": "p2",
                    "relative_hdf5_path": "TRAIN_shards/p2.h5",
                    "row_in_shard": 0,
                    "label": 1,
                    "filename": "p2-a.png",
                },
            ]
        ),
        shard_dir / "sample_manifest.parquet",
    )
    return lr_data.ShardDatasetLayout(
        shard_dir=shard_dir,
        manifest_path=shard_dir / "manifest.parquet",
        sample_manifest_path=shard_dir / "sample_manifest.parquet",
        local_cache_dir=None,
    )


class _IdentityTransform:
    def __call__(
        self,
        *,
        image: npt.NDArray[np.generic],
        mask: npt.NDArray[np.generic],
    ) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(np.moveaxis(image, -1, 0)),
            "mask": torch.from_numpy(mask),
        }


def test_setup_lr_finder_data_uses_filtered_train_shards_when_enabled(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_split_shards(data_root, "TRAIN_FILTERED_shards", ["p1"])
    _write_split_shards(data_root, "VALIDATION_shards", ["v1"])
    config = _build_config(tmp_path, smart_sampling=True)

    source_layout, validation_layout, source_split_name = lr_data.setup_lr_finder_data(config)

    assert source_split_name == "TRAIN_FILTERED_shards"
    assert source_layout.shard_dir == data_root / "TRAIN_FILTERED_shards"
    assert validation_layout.shard_dir == data_root / "VALIDATION_shards"


def test_setup_lr_finder_data_falls_back_to_train_shards(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_split_shards(data_root, "TRAIN_shards", ["p1"])
    _write_split_shards(data_root, "VALIDATION_shards", ["v1"])
    config = _build_config(tmp_path, smart_sampling=True)

    source_layout, _validation_layout, source_split_name = lr_data.setup_lr_finder_data(config)

    assert source_split_name == "TRAIN_shards"
    assert source_layout.shard_dir == data_root / "TRAIN_shards"


def test_training_shard_dataset_reads_sample_and_patient_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_split_shards(tmp_path / "data", "TRAIN_shards", ["p1", "p2"])
    monkeypatch.setattr(lr_data, "get_transforms", lambda mode, img_size: _IdentityTransform())

    dataset = lr_data.TrainingShardDataset(layout)
    image, mask = dataset[1]

    assert tuple(image.shape) == (3, 4, 4)
    assert tuple(mask.shape) == (4, 4)
    assert dataset.get_labels().tolist() == [0, 1]
    assert dataset.get_patient_ids().tolist() == ["p1", "p2"]


def test_training_shard_dataset_uses_row_in_shard_for_multirow_patient_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _write_multirow_train_shards(tmp_path / "data")
    monkeypatch.setattr(lr_data, "get_transforms", lambda mode, img_size: _IdentityTransform())

    dataset = lr_data.TrainingShardDataset(layout)
    first_image, first_mask = dataset[0]
    second_image, second_mask = dataset[1]
    third_image, third_mask = dataset[2]

    assert int(first_image[0, 0, 0]) == 22
    assert int(second_image[0, 0, 0]) == 11
    assert int(third_image[0, 0, 0]) == 33
    assert int(first_mask[0, 0]) == 1
    assert int(second_mask[0, 0]) == 0
    assert int(third_mask[0, 0]) == 2
    assert dataset.get_labels().tolist() == [1, 0, 1]
    assert dataset.get_patient_ids().tolist() == ["p1", "p1", "p2"]


def test_prepare_training_data_builds_weights_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    _write_split_shards(data_root, "TRAIN_shards", ["p1", "p2"])
    _write_split_shards(data_root, "VALIDATION_shards", ["v1"])
    config = _build_config(tmp_path, smart_sampling=False)
    monkeypatch.setattr(lr_data, "get_transforms", lambda mode, img_size: _IdentityTransform())

    prepared = lr_data.prepare_training_data(config)

    assert prepared.source_split_name == "TRAIN_shards"
    assert prepared.sample_weights.tolist() == [1.0, 1.0]
    assert prepared.training_provenance["manifest_path"].endswith("TRAIN_shards/manifest.parquet")
    assert prepared.validation_provenance["manifest_path"].endswith(
        "VALIDATION_shards/manifest.parquet"
    )


def test_build_train_loader_uses_weighted_sampler(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    layout = _write_split_shards(data_root, "TRAIN_shards", ["p1", "p2"])
    dataset = lr_data.TrainingShardDataset(layout, subset_indices=[0])
    sample_weights = torch.tensor([2.0], dtype=torch.float32)

    loader = lr_data.build_train_loader(dataset, sample_weights, batch_size=1, workers=0, seed=24)

    assert isinstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
    assert loader.worker_init_fn is lr_data.worker_init_fn  # type: ignore[attr-defined]
