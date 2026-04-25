from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from _pytest.monkeypatch import MonkeyPatch

from helpers.crossfold import normalization

MEDIAN_RGB_VALUE = 50


class _FakeRuntimeModule:
    def __init__(self, method: str) -> None:
        self.method = method

    def fit(self, target: torch.Tensor) -> None:
        assert target.shape == (1, 3, 2, 2)
        if self.method == "reinhard":
            self.target_means = torch.tensor([[[[0.1]], [[0.2]], [[0.3]]]])
            self.target_stds = torch.tensor([[[[0.4]], [[0.5]], [[0.6]]]])
            return
        self.stain_matrix_target = torch.tensor([[[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]]])
        self.maxC_target = torch.tensor([[1.0, 0.8]])


class _FakeRuntimeBuilder:
    calls: list[dict[str, Any]] = []

    @staticmethod
    def build(method: str, **kwargs: Any) -> _FakeRuntimeModule:
        _FakeRuntimeBuilder.calls.append({"method": method, **kwargs})
        return _FakeRuntimeModule(method)


def test_select_template_rows_from_entropy_picks_one_highest_entropy_row_per_patient() -> None:
    train_df = pd.DataFrame(
        [
            {"patient_id": 1, "image_path": "p1_low.png"},
            {"patient_id": 1, "image_path": "p1_high.png"},
            {"patient_id": 2, "image_path": "p2_only.png"},
        ]
    )
    entropy_df = pd.DataFrame(
        [
            {"image_path": "p1_low.png", "entropy": 0.1},
            {"image_path": "p1_high.png", "entropy": 0.9},
            {"image_path": "p2_only.png", "entropy": 0.4},
        ]
    )

    selected_rows = normalization.select_template_rows_from_entropy(train_df, entropy_df)

    assert selected_rows[["patient_id", "image_path", "entropy"]].to_dict("records") == [
        {"patient_id": 1, "image_path": "p1_high.png", "entropy": 0.9},
        {"patient_id": 2, "image_path": "p2_only.png", "entropy": 0.4},
    ]


def test_save_template_selection_artifacts_writes_shared_sidecars(tmp_path: Path) -> None:
    template_file = tmp_path / "template.png"
    cv2.imwrite(str(template_file), np.full((2, 2, 3), 33, dtype=np.uint8))
    train_df = pd.DataFrame([{"patient_id": 1, "image_path": str(template_file)}])
    entropy_df = pd.DataFrame([{"image_path": str(template_file), "entropy": 0.7}])
    selected_rows = pd.DataFrame(
        [{"patient_id": 1, "image_path": str(template_file), "entropy": 0.7}]
    )

    metadata = normalization.save_template_selection_artifacts(
        tmp_path,
        train_df=train_df,
        entropy_df=entropy_df,
        selected_rows=selected_rows,
        aggregate_target_rgb=np.full((2, 2, 3), 44, dtype=np.uint8),
        save_entropy_cache_csv=True,
    )

    assert metadata["selection_method"] == "highest_entropy_per_train_patient"
    assert metadata["template_count"] == 1
    assert (tmp_path / "entropy_cache.csv").is_file()
    assert (tmp_path / "patient_entropy_median.csv").is_file()
    assert (tmp_path / "aggregate_target.png").is_file()
    assert (tmp_path / "template_selection.json").is_file()
    assert (tmp_path / "template_selection" / "template_000_template.png").is_file()


def test_save_template_selection_artifacts_exports_hdf5_templates(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.full((1, 2, 2, 3), 80, dtype=np.uint8))
    train_df = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "image_path": f"{source_path}::images[0]",
                "source_hdf5_path": str(source_path),
                "source_row_index": 0,
            }
        ]
    )
    entropy_df = pd.DataFrame([{"image_path": f"{source_path}::images[0]", "entropy": 0.7}])
    selected_rows = train_df.assign(entropy=0.7)

    normalization.save_template_selection_artifacts(
        tmp_path,
        train_df=train_df,
        entropy_df=entropy_df,
        selected_rows=selected_rows,
        aggregate_target_rgb=np.full((2, 2, 3), 44, dtype=np.uint8),
        save_entropy_cache_csv=False,
    )

    exported = tmp_path / "template_selection" / "template_000_SOURCE_DATASET_row_000000.png"
    assert exported.is_file()


def test_generate_all_normalization_artifacts_uses_runtime_backend(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    template_dir = tmp_path / "template_selection"
    template_dir.mkdir()
    _FakeRuntimeBuilder.calls = []
    monkeypatch.setattr(
        normalization,
        "_load_torch_staintools_builder",
        lambda: _FakeRuntimeBuilder,
    )

    artifact_records = normalization.generate_all_normalization_artifacts(
        tmp_path,
        aggregate_target_rgb=np.full((2, 2, 3), 180, dtype=np.uint8),
        shared_template_dir=template_dir,
    )

    assert [record["method"] for record in artifact_records] == [
        "REINHARD",
        "RUIFROK",
        "MACENKO",
        "VAHADANE",
    ]
    assert [call["method"] for call in _FakeRuntimeBuilder.calls] == [
        "reinhard",
        "macenko",
        "vahadane",
    ]
    assert all(call["use_cache"] is False for call in _FakeRuntimeBuilder.calls)
    reinhard_stats = (
        tmp_path / "runtime_normalization_artifacts" / "reinhard" / "normalization_stats.json"
    )
    macenko_stats = (
        tmp_path / "runtime_normalization_artifacts" / "macenko" / "normalization_stats.json"
    )
    ruifrok_stats = (
        tmp_path / "runtime_normalization_artifacts" / "ruifrok" / "normalization_stats.json"
    )
    assert "target_means" in reinhard_stats.read_text(encoding="utf-8")
    assert "stain_matrix_target" in macenko_stats.read_text(encoding="utf-8")
    assert "stain_matrix_source" in ruifrok_stats.read_text(encoding="utf-8")


def test_make_aggregate_target_uses_median_rgb(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    cv2.imwrite(str(image_a), np.zeros((2, 2, 3), dtype=np.uint8))
    cv2.imwrite(str(image_b), np.full((2, 2, 3), 100, dtype=np.uint8))

    target = normalization.make_aggregate_target([str(image_a), str(image_b)])

    assert target.shape == (2, 2, 3)
    assert int(target[0, 0, 0]) == MEDIAN_RGB_VALUE


def test_build_aggregate_target_from_hdf5_template_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.stack(
                [
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    np.full((2, 2, 3), 100, dtype=np.uint8),
                ],
                axis=0,
            ),
        )
    selected_rows = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "image_path": f"{source_path}::images[0]",
                "source_hdf5_path": str(source_path),
                "source_row_index": 0,
            },
            {
                "patient_id": 2,
                "image_path": f"{source_path}::images[1]",
                "source_hdf5_path": str(source_path),
                "source_row_index": 1,
            },
        ]
    )

    target = normalization.build_aggregate_target_from_template_rows(selected_rows)

    assert target.shape == (2, 2, 3)
    assert int(target[0, 0, 0]) == MEDIAN_RGB_VALUE
