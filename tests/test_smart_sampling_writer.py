from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

from helpers.smart_sampling.config import SmartSamplerConfig
from helpers.smart_sampling.writer import write_filtered_hdf5


def _write_source_hdf5(path: Path, *, filename_dataset_name: str = "filenames") -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
        handle.create_dataset("masks", data=np.zeros((3, 4, 4), dtype=np.uint8))
        handle.create_dataset("patient_ids", data=np.array([1, 1, 2], dtype=np.int32))
        handle.create_dataset("labels", data=np.array([0, 1, 1], dtype=np.uint8))
        handle.create_dataset(
            filename_dataset_name,
            data=np.array(
                [b"PATIENT_1_a.png", b"PATIENT_1_b.png", b"PATIENT_2_a.png"],
                dtype="S32",
            ),
        )


def _build_config(source_path: Path, output_dir: Path, **overrides: object) -> SmartSamplerConfig:
    values: dict[str, Any] = {
        "source_h5_path": source_path,
        "output_dir": output_dir,
        "output_filename": "TRAIN_FILTERED.h5",
        "local_work_dir": None,
        "stage_input_locally": False,
        "stage_outputs_locally": False,
        "clean_local_work_dir": True,
        "write_sidecars": True,
        "overwrite_output": True,
        "model_name": "owkin/phikon-v2",
        "batch_size": 8,
        "device": "cpu",
        "n_start": 8,
        "n_max": 8,
        "growth_factor": 2.0,
        "stability_threshold": 0.85,
        "stability_repeats": 2,
        "max_steps": 2,
        "intersection_ratio_threshold": 0.2,
        "k_min": 20,
        "k_max": 80,
        "adaptive_keep_enabled": True,
        "keep_min": 2,
        "keep_step": 1,
        "keep_improvement_threshold": 0.02,
        "keep_patience": 2,
        "m_max": 2,
        "seed": 42,
        "num_workers": 0,
    }
    values.update(overrides)
    return SmartSamplerConfig(**values)


def test_write_filtered_hdf5_propagates_upstream_lineage_attrs(tmp_path: Path) -> None:
    source_path = tmp_path / "TRAIN.h5"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_source_hdf5(source_path)
    with h5py.File(source_path, "a") as handle:
        handle.attrs["source_signature"] = "stage5-signature"
        handle.attrs["upstream_source_signature"] = "stage2-signature"
        handle.attrs["stage4_cleaning_manifest_path"] = "/tmp/accepted_manifest.csv"
        handle.attrs["stage4_cleaning_manifest_sha256"] = "abc123"
        handle.attrs["stage4_cleaning_selected_rows"] = 3

    config = _build_config(source_path, output_dir)

    result = write_filtered_hdf5(config, np.array([1, 2], dtype=np.int64))

    with h5py.File(result, "r") as handle:
        assert handle.attrs["source_signature"] == "stage5-signature"
        assert handle.attrs["upstream_source_signature"] == "stage2-signature"
        assert handle.attrs["stage4_cleaning_manifest_path"] == "/tmp/accepted_manifest.csv"
        assert handle.attrs["stage4_cleaning_manifest_sha256"] == "abc123"
        assert handle.attrs["stage4_cleaning_selected_rows"] == 3


def test_write_filtered_hdf5_writes_plural_filenames_for_legacy_input(tmp_path: Path) -> None:
    source_path = tmp_path / "TRAIN.h5"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_source_hdf5(source_path, filename_dataset_name="filename")

    config = _build_config(source_path, output_dir)

    result = write_filtered_hdf5(config, np.array([1, 2], dtype=np.int64))

    with h5py.File(result, "r") as handle:
        filenames = cast(h5py.Dataset, handle["filenames"])
        patient_ids = cast(h5py.Dataset, handle["patient_ids"])
        assert set(handle.keys()) == {"filenames", "images", "labels", "masks", "patient_ids"}
        assert handle.attrs["selection_signature"]
        assert cast(bytes, filenames[0]).decode("utf-8") == "PATIENT_1_b.png"
        assert patient_ids[:].tolist() == [1, 2]


def test_write_filtered_hdf5_reuses_existing_output_when_selection_matches(tmp_path: Path) -> None:
    source_path = tmp_path / "TRAIN.h5"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_source_hdf5(source_path)

    config = _build_config(source_path, output_dir)
    write_filtered_hdf5(config, np.array([1, 2], dtype=np.int64))
    reuse_config = SmartSamplerConfig(**{**config.__dict__, "overwrite_output": False})

    result = write_filtered_hdf5(reuse_config, np.array([1, 2], dtype=np.int64))

    assert result == output_dir / "TRAIN_FILTERED.h5"


def test_write_filtered_hdf5_rejects_existing_output_when_selection_mismatches(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "TRAIN.h5"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_source_hdf5(source_path)

    config = _build_config(source_path, output_dir)
    write_filtered_hdf5(config, np.array([1, 2], dtype=np.int64))
    reuse_config = SmartSamplerConfig(**{**config.__dict__, "overwrite_output": False})

    try:
        write_filtered_hdf5(reuse_config, np.array([0, 2], dtype=np.int64))
    except ValueError as error:
        assert "does not match the current selection" in str(error)
    else:
        raise AssertionError("Expected stale smart-sampling output to be rejected.")


def test_write_filtered_hdf5_rejects_reuse_when_source_contents_change_in_place(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "TRAIN.h5"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_source_hdf5(source_path)

    config = _build_config(source_path, output_dir)
    write_filtered_hdf5(config, np.array([1, 2], dtype=np.int64))
    reuse_config = SmartSamplerConfig(**{**config.__dict__, "overwrite_output": False})

    with h5py.File(source_path, "a") as handle:
        labels = cast(h5py.Dataset, handle["labels"])
        labels[0] = 1

    with pytest.raises(ValueError, match="does not match the current selection"):
        write_filtered_hdf5(reuse_config, np.array([1, 2], dtype=np.int64))


def test_write_filtered_hdf5_keeps_selection_signature_stable_when_input_is_staged(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "TRAIN.h5"
    output_dir = tmp_path / "output"
    staged_dir = tmp_path / "content"
    output_dir.mkdir()
    staged_dir.mkdir()
    _write_source_hdf5(source_path)
    staged_source_path = staged_dir / source_path.name
    staged_source_path.write_bytes(source_path.read_bytes())

    config = _build_config(
        source_path,
        output_dir,
        local_work_dir=staged_dir,
        stage_input_locally=True,
        stage_outputs_locally=False,
    )

    direct_output = write_filtered_hdf5(config, np.array([1, 2], dtype=np.int64))
    direct_signature = None
    with h5py.File(direct_output, "r") as handle:
        direct_signature = handle.attrs["selection_signature"]

    staged_output = write_filtered_hdf5(
        config,
        np.array([1, 2], dtype=np.int64),
        source_h5_path=staged_source_path,
        output_dir=staged_dir,
        signature_source_path=source_path,
    )
    with h5py.File(staged_output, "r") as handle:
        staged_signature = handle.attrs["selection_signature"]

    assert staged_signature == direct_signature
