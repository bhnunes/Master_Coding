import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
from _pytest.monkeypatch import MonkeyPatch

from helpers.crossfold import normalization


class _FakeNormalizer:
    def __init__(self) -> None:
        self.fitted_target: Any = None

    def fit(self, target: object) -> None:
        self.fitted_target = target


class _FakeStainModule:
    def __init__(self, normalizer: _FakeNormalizer) -> None:
        self._normalizer = normalizer

    def get_normalizer(self, method_name: str) -> _FakeNormalizer:
        assert method_name == "MACENKO"
        return self._normalizer


def test_fit_normalizer_on_train_set_uses_highest_entropy_image_per_patient(
    monkeypatch: MonkeyPatch,
) -> None:
    selected_targets: list[list[str]] = []
    fake_normalizer = _FakeNormalizer()
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

    monkeypatch.setattr(
        normalization,
        "load_stain_normalizer_backend",
        lambda: _FakeStainModule(fake_normalizer),
    )
    monkeypatch.setattr(
        normalization,
        "make_aggregate_target",
        lambda image_paths: _record_target(selected_targets, image_paths),
    )

    fitted_normalizer, template_paths = normalization.fit_normalizer_on_train_set(
        train_df,
        "MACENKO",
        entropy_df=entropy_df,
    )

    assert fitted_normalizer is fake_normalizer
    assert template_paths == ["p1_high.png", "p2_only.png"]
    assert selected_targets == [["p1_high.png", "p2_only.png"]]
    assert fake_normalizer.fitted_target == "aggregate-target"


def _record_target(selected_targets: list[list[str]], image_paths: list[str]) -> str:
    selected_targets.append(list(image_paths))
    return "aggregate-target"


def test_save_normalizer_stats_writes_json_and_template_copies(tmp_path: Path) -> None:
    template_file = tmp_path / "template.png"
    template_file.write_bytes(b"image")
    fake_module = type(
        "FakeStainModule",
        (),
        {
            "StainNormalizer": type("FakeStainNormalizer", (), {}),
            "ReinhardNormalizer": type("FakeReinhardNormalizer", (), {}),
        },
    )

    normalization.save_normalizer_stats(
        normalizer=object(),
        method_name="MACENKO",
        output_dir=tmp_path,
        template_paths=[str(template_file)],
        stainnorm_module=fake_module,
    )

    assert (tmp_path / "normalization_stats.json").is_file()
    assert (tmp_path / "normalization_templates" / "template_000_template.png").is_file()


def test_make_aggregate_target_uses_median_rgb(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    cv2.imwrite(str(image_a), np.zeros((2, 2, 3), dtype=np.uint8))
    cv2.imwrite(str(image_b), np.full((2, 2, 3), 100, dtype=np.uint8))

    target = normalization.make_aggregate_target([str(image_a), str(image_b)])

    assert target.shape == (2, 2, 3)
    assert int(target[0, 0, 0]) == 50


def test_fit_normalizer_on_train_set_falls_back_to_rereading_images(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_normalizer = _FakeNormalizer()
    train_df = pd.DataFrame(
        [
            {"patient_id": 1, "image_path": "a.png"},
            {"patient_id": 1, "image_path": "b.png"},
        ]
    )

    monkeypatch.setattr(
        normalization,
        "calculate_image_entropy_from_path",
        lambda path: (path, {"a.png": 0.1, "b.png": 0.8}[path]),
    )
    monkeypatch.setattr(
        normalization,
        "load_stain_normalizer_backend",
        lambda: _FakeStainModule(fake_normalizer),
    )
    monkeypatch.setattr(normalization, "make_aggregate_target", lambda image_paths: image_paths)

    _, template_paths = normalization.fit_normalizer_on_train_set(
        train_df, "MACENKO", entropy_df=None
    )

    assert template_paths == ["b.png"]
    assert fake_normalizer.fitted_target == ["b.png"]


def test_load_stain_normalizer_backend_uses_tiatoolbox_module(monkeypatch: MonkeyPatch) -> None:
    fake_stainnorm = object()
    fake_tools_module = type("FakeToolsModule", (), {"stainnorm": fake_stainnorm})

    monkeypatch.setattr(normalization, "load_openslide_module", lambda: object())
    monkeypatch.setitem(sys.modules, "tiatoolbox.tools", fake_tools_module)

    assert normalization.load_stain_normalizer_backend() is fake_stainnorm


def test_fit_normalizer_on_train_set_supports_hdf5_backed_template_rows(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_normalizer = _FakeNormalizer()
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset(
            "images",
            data=np.stack(
                [
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    np.full((2, 2, 3), 25, dtype=np.uint8),
                    np.full((2, 2, 3), 50, dtype=np.uint8),
                ],
                axis=0,
            ),
        )

    train_df = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "image_path": f"{source_path}::images[0]",
                "source_hdf5_path": str(source_path),
                "source_row_index": 0,
            },
            {
                "patient_id": 1,
                "image_path": f"{source_path}::images[1]",
                "source_hdf5_path": str(source_path),
                "source_row_index": 1,
            },
            {
                "patient_id": 2,
                "image_path": f"{source_path}::images[2]",
                "source_hdf5_path": str(source_path),
                "source_row_index": 2,
            },
        ]
    )
    entropy_df = pd.DataFrame(
        [
            {"image_path": f"{source_path}::images[0]", "entropy": 0.1},
            {"image_path": f"{source_path}::images[1]", "entropy": 0.9},
            {"image_path": f"{source_path}::images[2]", "entropy": 0.4},
        ]
    )

    monkeypatch.setattr(
        normalization,
        "load_stain_normalizer_backend",
        lambda: _FakeStainModule(fake_normalizer),
    )

    fitted_normalizer, template_paths = normalization.fit_normalizer_on_train_set(
        train_df,
        "MACENKO",
        entropy_df=entropy_df,
    )

    assert fitted_normalizer is fake_normalizer
    assert template_paths == [f"{source_path}::images[1]", f"{source_path}::images[2]"]
    assert np.array_equal(fake_normalizer.fitted_target, np.full((2, 2, 3), 37, dtype=np.uint8))


def test_save_normalizer_stats_exports_hdf5_template_images(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE_DATASET.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("images", data=np.full((1, 2, 2, 3), 80, dtype=np.uint8))

    fake_module = type(
        "FakeStainModule",
        (),
        {
            "StainNormalizer": type("FakeStainNormalizer", (), {}),
            "ReinhardNormalizer": type("FakeReinhardNormalizer", (), {}),
        },
    )

    normalization.save_normalizer_stats(
        normalizer=object(),
        method_name="MACENKO",
        output_dir=tmp_path,
        template_paths=[f"{source_path}::images[0]"],
        stainnorm_module=fake_module,
    )

    exported = tmp_path / "normalization_templates" / "template_000_SOURCE_DATASET_row_000000.png"
    assert exported.is_file()


def test_save_normalizer_stats_extracts_known_normalizer_fields(tmp_path: Path) -> None:
    template_file = tmp_path / "template.png"
    template_file.write_bytes(b"image")

    class FakeStainNormalizer:
        def __init__(self) -> None:
            self.stain_matrix_target = np.array([[1.0, 2.0]])
            self.maxC_target = np.array([3.0])
            self.extractor = type("Extractor", (), {"stains": np.array([[4.0, 5.0]])})()

    fake_module = type(
        "FakeStainModule",
        (),
        {
            "StainNormalizer": FakeStainNormalizer,
            "ReinhardNormalizer": type("FakeReinhard", (), {}),
        },
    )

    normalization.save_normalizer_stats(
        normalizer=FakeStainNormalizer(),
        method_name="MACENKO",
        output_dir=tmp_path,
        template_paths=[str(template_file)],
        stainnorm_module=fake_module,
    )

    text = (tmp_path / "normalization_stats.json").read_text(encoding="utf-8")
    assert "stain_matrix_target" in text
    assert "maxC_target" in text
