from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Hashable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from helpers.extraction.manifest_paths import to_manifest_path_ref
from helpers.provenance import hash_file_sha256
from helpers.training import stain_normalization as stain_norm
from helpers.training.master_manifest_queries import CanonicalRowRecord
from helpers.training.stain_normalization import (
    build_split_stain_normalizer,
    resolve_dataloader_stain_normalizer_device,
)

NORMALIZED_PIXEL_VALUE = 11


class _FakeRuntimeNormalizer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        x: torch.Tensor,
        cache_keys: list[str] | None = None,
    ) -> torch.Tensor:
        self.calls.append({"shape": tuple(x.shape), "cache_keys": cache_keys})
        return x + (1.0 / 255.0)


class _FakeTensorCache:
    def __init__(self) -> None:
        self.values: dict[Hashable, torch.Tensor] = {}

    def __contains__(self, key: Hashable) -> bool:
        return key in self.values

    def query(self, key: Hashable) -> torch.Tensor:
        return self.values[key]

    def write_to_cache(self, key: Hashable, value: torch.Tensor) -> None:
        self.values[key] = value.detach().clone()


class _FakeCachingRuntimeNormalizer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tensor_cache = _FakeTensorCache()
        self.source_fit_calls = 0

    def forward(
        self,
        x: torch.Tensor,
        cache_keys: list[Hashable] | None = None,
    ) -> torch.Tensor:
        cache_key = cache_keys[0] if cache_keys is not None else "uncached"
        if cache_key not in self.tensor_cache:
            self.source_fit_calls += 1
            self.tensor_cache.write_to_cache(
                cache_key,
                torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]], dtype=torch.float32),
            )
        return x + (1.0 / 255.0)


class _FakeNormalizerBuilder:
    last_method: str | None = None
    last_kwargs: dict[str, Any] | None = None
    last_module: _FakeRuntimeNormalizer | None = None

    @staticmethod
    def build(method: str, **kwargs: Any) -> _FakeRuntimeNormalizer:
        _FakeNormalizerBuilder.last_method = method
        _FakeNormalizerBuilder.last_kwargs = kwargs
        module = _FakeRuntimeNormalizer()
        _FakeNormalizerBuilder.last_module = module
        return module


class _FakeCachingNormalizerBuilder:
    instances: list[_FakeCachingRuntimeNormalizer] = []

    @staticmethod
    def build(method: str, **kwargs: Any) -> _FakeCachingRuntimeNormalizer:
        del method, kwargs
        module = _FakeCachingRuntimeNormalizer()
        _FakeCachingNormalizerBuilder.instances.append(module)
        return module


def test_resolve_dataloader_stain_normalizer_device_uses_cpu_for_cuda_workers() -> None:
    device = resolve_dataloader_stain_normalizer_device("cuda", workers=2)

    assert device == torch.device("cpu")


def test_resolve_dataloader_stain_normalizer_device_keeps_requested_device_without_workers() -> (
    None
):
    device = resolve_dataloader_stain_normalizer_device("cuda", workers=0)

    assert device == torch.device("cuda")


def test_load_torch_staintools_builder_disables_compile_for_runtime_normalization() -> None:
    pytest.importorskip("torch_staintools")
    from torch_staintools.constants import CONFIG

    original_enable_compile = CONFIG.ENABLE_COMPILE
    try:
        CONFIG.ENABLE_COMPILE = True

        builder = stain_norm._load_torch_staintools_builder()

        assert builder is not None
        assert CONFIG.ENABLE_COMPILE is False
    finally:
        CONFIG.ENABLE_COMPILE = original_enable_compile


def _write_normalization_manifest(
    master_manifest_path: Path,
    *,
    method: str,
    state_path: Path,
    state_sha256: str,
    stage4_split_bundle_id: int = 1,
) -> None:
    with sqlite3.connect(master_manifest_path) as connection:
        connection.executescript(
            """
            CREATE TABLE stage4_split_bundles (
                stage4_split_bundle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                output_dir_path TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE normalization_artifacts (
                normalization_artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                state_path TEXT NOT NULL,
                state_sha256 TEXT NOT NULL,
                template_path TEXT,
                template_sha256 TEXT,
                fit_scope TEXT NOT NULL
            );
            CREATE TABLE stage4_split_bundle_artifacts (
                stage4_split_bundle_id INTEGER NOT NULL,
                normalization_artifact_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (stage4_split_bundle_id, normalization_artifact_id)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO stage4_split_bundles (stage4_split_bundle_id, run_id, output_dir_path)
            VALUES (?, 1, 'MANIFEST::stage4')
            """,
            (stage4_split_bundle_id,),
        )
        connection.execute(
            """
            INSERT INTO normalization_artifacts (
                run_id,
                method,
                state_path,
                state_sha256,
                template_path,
                template_sha256,
                fit_scope
            ) VALUES (1, ?, ?, ?, NULL, NULL, 'TRAIN')
            """,
            (
                method,
                to_manifest_path_ref(state_path, manifest_path=master_manifest_path),
                state_sha256,
            ),
        )
        connection.execute(
            """
            INSERT INTO stage4_split_bundle_artifacts (
                stage4_split_bundle_id,
                normalization_artifact_id
            ) VALUES (?, 1)
            """,
            (stage4_split_bundle_id,),
        )
        connection.commit()


def _write_shared_bundle_normalization_manifest(
    master_manifest_path: Path,
    *,
    artifacts: list[tuple[str, Path]],
    stage4_split_bundle_id: int = 1,
) -> None:
    with sqlite3.connect(master_manifest_path) as connection:
        connection.executescript(
            """
            CREATE TABLE stage4_split_bundles (
                stage4_split_bundle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                output_dir_path TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE normalization_artifacts (
                normalization_artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                state_path TEXT NOT NULL,
                state_sha256 TEXT NOT NULL,
                template_path TEXT,
                template_sha256 TEXT,
                fit_scope TEXT NOT NULL
            );
            CREATE TABLE stage4_split_bundle_artifacts (
                stage4_split_bundle_id INTEGER NOT NULL,
                normalization_artifact_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (stage4_split_bundle_id, normalization_artifact_id)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO stage4_split_bundles (stage4_split_bundle_id, run_id, output_dir_path)
            VALUES (?, 1, 'MANIFEST::stage4')
            """,
            (stage4_split_bundle_id,),
        )
        for method, state_path in artifacts:
            cursor = connection.execute(
                """
                INSERT INTO normalization_artifacts (
                    run_id,
                    method,
                    state_path,
                    state_sha256,
                    template_path,
                    template_sha256,
                    fit_scope
                ) VALUES (1, ?, ?, ?, NULL, NULL, 'TRAIN')
                """,
                (
                    method,
                    to_manifest_path_ref(state_path, manifest_path=master_manifest_path),
                    hash_file_sha256(state_path),
                ),
            )
            connection.execute(
                """
                INSERT INTO stage4_split_bundle_artifacts (
                    stage4_split_bundle_id,
                    normalization_artifact_id
                ) VALUES (?, ?)
                """,
                (stage4_split_bundle_id, int(cursor.lastrowid or 0)),
            )
        connection.commit()


def _make_record(
    *,
    stage4_split_bundle_id: int | None = 1,
) -> CanonicalRowRecord:
    return CanonicalRowRecord(
        source_hdf5_path=Path("/tmp/source.h5"),
        source_row_index=0,
        patient_id="1",
        label=1,
        filename="patch_001.png",
        split="TRAIN",
        normalization_method=None,
        normalization_artifact_id=None,
        sampling_decision=None,
        is_stage7_selected=True,
        stage4_split_bundle_id=stage4_split_bundle_id,
    )


def test_build_split_stain_normalizer_returns_none_for_not_normalized(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    state_path = tmp_path / "unused.json"
    state_path.write_text("{}", encoding="utf-8")
    _write_normalization_manifest(
        master_manifest_path,
        method="REINHARD",
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )

    normalizer = build_split_stain_normalizer(
        master_manifest_path,
        [_make_record()],
        runtime_normalization_method="NOT_NORMALIZED",
    )

    assert normalizer is None


def test_build_split_stain_normalizer_rejects_legacy_absolute_state_path(
    tmp_path: Path,
) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    state_path = tmp_path / "normalization_stats.json"
    state_path.write_text("{}", encoding="utf-8")

    with sqlite3.connect(master_manifest_path) as connection:
        connection.executescript(
            """
            CREATE TABLE stage4_split_bundles (
                stage4_split_bundle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                output_dir_path TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE normalization_artifacts (
                normalization_artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                state_path TEXT NOT NULL,
                state_sha256 TEXT NOT NULL,
                template_path TEXT,
                template_sha256 TEXT,
                fit_scope TEXT NOT NULL
            );
            CREATE TABLE stage4_split_bundle_artifacts (
                stage4_split_bundle_id INTEGER NOT NULL,
                normalization_artifact_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (stage4_split_bundle_id, normalization_artifact_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO stage4_split_bundles (stage4_split_bundle_id, run_id, output_dir_path) "
            "VALUES (1, 1, 'MANIFEST::stage4')"
        )
        connection.execute(
            """
            INSERT INTO normalization_artifacts (
                run_id,
                method,
                state_path,
                state_sha256,
                template_path,
                template_sha256,
                fit_scope
            ) VALUES (1, 'REINHARD', ?, ?, NULL, NULL, 'TRAIN')
            """,
            (str(state_path), hash_file_sha256(state_path)),
        )
        connection.execute(
            "INSERT INTO stage4_split_bundle_artifacts "
            "(stage4_split_bundle_id, normalization_artifact_id) "
            "VALUES (1, 1)"
        )
        connection.commit()

    with pytest.raises(ValueError, match="Legacy absolute-path manifest values are not supported"):
        build_split_stain_normalizer(
            master_manifest_path,
            [_make_record()],
            runtime_normalization_method="REINHARD",
        )


def test_build_split_stain_normalizer_resolves_state_path_after_manifest_relocation(
    tmp_path: Path,
) -> None:
    original_dir = tmp_path / "local_run"
    relocated_dir = tmp_path / "colab_run"
    original_dir.mkdir()
    relocated_dir.mkdir()

    original_manifest_path = original_dir / "master_manifest.sqlite"
    original_state_path = original_dir / "normalization_stats.json"
    original_state_path.write_text(
        json.dumps(
            {
                "target_means": [0.1, 0.2, 0.3],
                "target_stds": [0.4, 0.5, 0.6],
            }
        ),
        encoding="utf-8",
    )
    _write_normalization_manifest(
        original_manifest_path,
        method="REINHARD",
        state_path=original_state_path,
        state_sha256=hash_file_sha256(original_state_path),
    )

    relocated_manifest_path = relocated_dir / "master_manifest.sqlite"
    relocated_state_path = relocated_dir / "normalization_stats.json"
    shutil.copy2(original_manifest_path, relocated_manifest_path)
    shutil.copy2(original_state_path, relocated_state_path)

    normalizer = build_split_stain_normalizer(
        relocated_manifest_path,
        [_make_record()],
        runtime_normalization_method="REINHARD",
    )

    assert normalizer is not None


def test_build_split_stain_normalizer_loads_reinhard_state_and_normalizes_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "normalization_stats.json"
    state_path.write_text(
        json.dumps(
            {
                "method": "REINHARD",
                "target_means": [0.1, 0.2, 0.3],
                "target_stds": [0.4, 0.5, 0.6],
            }
        ),
        encoding="utf-8",
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_normalization_manifest(
        master_manifest_path,
        method="REINHARD",
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )
    monkeypatch.setattr(
        "helpers.training.stain_normalization._load_torch_staintools_builder",
        lambda: _FakeNormalizerBuilder,
    )

    normalizer = build_split_stain_normalizer(
        master_manifest_path,
        [_make_record()],
        runtime_normalization_method="REINHARD",
    )

    assert normalizer is not None
    image = np.full((4, 4, 3), 10, dtype=np.uint8)
    output = normalizer.normalize_image(image, cache_key="patch_001.png")

    assert _FakeNormalizerBuilder.last_method == "reinhard"
    assert _FakeNormalizerBuilder.last_kwargs is not None
    assert _FakeNormalizerBuilder.last_kwargs["use_cache"] is True
    assert np.all(output == NORMALIZED_PIXEL_VALUE)
    assert _FakeNormalizerBuilder.last_module is not None
    assert _FakeNormalizerBuilder.last_module.calls == [
        {"shape": (1, 3, 4, 4), "cache_keys": ["patch_001.png"]}
    ]
    target_means = cast(torch.Tensor, _FakeNormalizerBuilder.last_module.target_means)
    target_stds = cast(torch.Tensor, _FakeNormalizerBuilder.last_module.target_stds)
    assert torch.equal(
        target_means,
        torch.tensor([[[[0.1]], [[0.2]], [[0.3]]]], dtype=torch.float32),
    )
    assert torch.equal(
        target_stds,
        torch.tensor([[[[0.4]], [[0.5]], [[0.6]]]], dtype=torch.float32),
    )


def test_build_split_stain_normalizer_fails_closed_on_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "normalization_stats.json"
    state_path.write_text(
        json.dumps({"target_means": [0.1], "target_stds": [0.2]}), encoding="utf-8"
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_normalization_manifest(
        master_manifest_path,
        method="REINHARD",
        state_path=state_path,
        state_sha256="deadbeef",
    )
    monkeypatch.setattr(
        "helpers.training.stain_normalization._load_torch_staintools_builder",
        lambda: _FakeNormalizerBuilder,
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        build_split_stain_normalizer(
            master_manifest_path,
            [_make_record()],
            runtime_normalization_method="REINHARD",
        )


def test_build_split_stain_normalizer_fails_when_state_file_is_missing(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "normalization_stats.json"
    state_path.write_text(
        json.dumps({"method": "REINHARD", "target_means": [0.1], "target_stds": [0.2]}),
        encoding="utf-8",
    )
    state_sha256 = hash_file_sha256(state_path)
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_normalization_manifest(
        master_manifest_path,
        method="REINHARD",
        state_path=state_path,
        state_sha256=state_sha256,
    )
    state_path.unlink()

    with pytest.raises(FileNotFoundError, match="Normalization state file is missing"):
        build_split_stain_normalizer(
            master_manifest_path,
            [_make_record()],
            runtime_normalization_method="REINHARD",
        )


def test_build_split_stain_normalizer_rejects_inconsistent_records(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    state_path = tmp_path / "unused.json"
    state_path.write_text("{}", encoding="utf-8")
    _write_normalization_manifest(
        master_manifest_path,
        method="REINHARD",
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )

    with pytest.raises(ValueError, match="do not agree"):
        build_split_stain_normalizer(
            master_manifest_path,
            [
                _make_record(stage4_split_bundle_id=1),
                _make_record(stage4_split_bundle_id=2),
            ],
            runtime_normalization_method="REINHARD",
        )


def test_build_split_stain_normalizer_rejects_invalid_runtime_method(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    state_path = tmp_path / "unused.json"
    state_path.write_text("{}", encoding="utf-8")
    _write_normalization_manifest(
        master_manifest_path,
        method="REINHARD",
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )

    with pytest.raises(ValueError, match="Unsupported runtime normalization method"):
        build_split_stain_normalizer(
            master_manifest_path,
            [_make_record()],
            runtime_normalization_method="INVALID_METHOD",
        )


def test_build_split_stain_normalizer_fails_when_requested_artifact_missing(tmp_path: Path) -> None:
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    state_path = tmp_path / "reinhard_stats.json"
    state_path.write_text(
        json.dumps({"method": "REINHARD", "target_means": [0.1], "target_stds": [0.2]}),
        encoding="utf-8",
    )
    _write_normalization_manifest(
        master_manifest_path,
        method="REINHARD",
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )

    with pytest.raises(ValueError, match="Requested runtime normalization artifact is missing"):
        build_split_stain_normalizer(
            master_manifest_path,
            [_make_record()],
            runtime_normalization_method="MACENKO",
        )


def test_build_split_stain_normalizer_supports_ruifrok(tmp_path: Path) -> None:
    state_path = tmp_path / "normalization_stats.json"
    state_path.write_text(
        json.dumps(
            {
                "method": "RUIFROK",
                "stain_matrix_target": [
                    [0.65, 0.70, 0.29],
                    [0.07, 0.99, 0.11],
                ],
                "maxC_target": [1.0, 0.8],
            }
        ),
        encoding="utf-8",
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_normalization_manifest(
        master_manifest_path,
        method="RUIFROK",
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )

    normalizer = build_split_stain_normalizer(
        master_manifest_path,
        [_make_record()],
        runtime_normalization_method="RUIFROK",
    )

    assert normalizer is not None
    image = np.full((3, 3, 3), 180, dtype=np.uint8)
    output = normalizer.normalize_image(image, cache_key="patch_001.png")
    assert output.shape == image.shape
    assert output.dtype == np.uint8


def test_build_split_stain_normalizer_uses_fixed_source_vahadane_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "vahadane_stats.json"
    target_matrix = [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]]
    state_path.write_text(
        json.dumps(
            {
                "method": "VAHADANE",
                "stain_matrix_target": target_matrix,
                "maxC_target": [1.0, 0.8],
            }
        ),
        encoding="utf-8",
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_normalization_manifest(
        master_manifest_path,
        method="VAHADANE",
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )
    monkeypatch.setattr(
        "helpers.training.stain_normalization._load_torch_staintools_builder",
        lambda: pytest.fail("fixed-source VAHADANE must not load torch-staintools"),
    )

    normalizer = build_split_stain_normalizer(
        master_manifest_path,
        [_make_record()],
        runtime_normalization_method="VAHADANE",
        device="cpu",
    )

    assert normalizer is not None
    module = cast(torch.nn.Module, cast(Any, normalizer).module)
    assert isinstance(module, stain_norm._FixedMatrixDeconvolutionNormalizer)
    source_matrix = cast(torch.Tensor, module.stain_matrix_source)
    target_matrix_tensor = cast(torch.Tensor, module.stain_matrix_target)
    expected_matrix = torch.tensor([target_matrix], dtype=torch.float32)
    assert torch.equal(source_matrix, expected_matrix)
    assert torch.equal(target_matrix_tensor, expected_matrix)


def test_build_split_stain_normalizer_uses_vahadane_source_matrix_when_available(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "vahadane_stats.json"
    source_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    target_matrix = [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]]
    state_path.write_text(
        json.dumps(
            {
                "method": "VAHADANE",
                "stain_matrix_source": source_matrix,
                "stain_matrix_target": target_matrix,
                "maxC_target": [1.0, 0.8],
            }
        ),
        encoding="utf-8",
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_normalization_manifest(
        master_manifest_path,
        method="VAHADANE",
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )

    normalizer = build_split_stain_normalizer(
        master_manifest_path,
        [_make_record()],
        runtime_normalization_method="VAHADANE",
        device="cpu",
    )

    assert normalizer is not None
    module = cast(torch.nn.Module, cast(Any, normalizer).module)
    assert torch.equal(
        cast(torch.Tensor, module.stain_matrix_source),
        torch.tensor([source_matrix], dtype=torch.float32),
    )


@pytest.mark.parametrize("method", ["MACENKO"])
def test_build_split_stain_normalizer_supports_torch_staintools_matrix_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    state_path = tmp_path / f"{method.lower()}_stats.json"
    state_path.write_text(
        json.dumps(
            {
                "method": method,
                "stain_matrix_target": [
                    [0.65, 0.70, 0.29],
                    [0.07, 0.99, 0.11],
                ],
                "maxC_target": [1.0, 0.8],
            }
        ),
        encoding="utf-8",
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_normalization_manifest(
        master_manifest_path,
        method=method,
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )
    monkeypatch.setattr(
        "helpers.training.stain_normalization._load_torch_staintools_builder",
        lambda: _FakeNormalizerBuilder,
    )

    normalizer = build_split_stain_normalizer(
        master_manifest_path,
        [_make_record()],
        runtime_normalization_method=method,
    )

    assert normalizer is not None
    image = np.full((4, 4, 3), 10, dtype=np.uint8)
    output = normalizer.normalize_image(image, cache_key="patch_001.png")

    assert _FakeNormalizerBuilder.last_method == method.lower()
    assert np.all(output == NORMALIZED_PIXEL_VALUE)
    assert _FakeNormalizerBuilder.last_module is not None
    stain_matrix_target = cast(torch.Tensor, _FakeNormalizerBuilder.last_module.stain_matrix_target)
    max_c_target = cast(torch.Tensor, _FakeNormalizerBuilder.last_module.maxC_target)
    assert torch.equal(
        stain_matrix_target,
        torch.tensor(
            [[[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(max_c_target, torch.tensor([[1.0, 0.8]], dtype=torch.float32))


def test_build_split_stain_normalizer_can_opt_into_exact_vahadane_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "vahadane_stats.json"
    state_path.write_text(
        json.dumps(
            {
                "method": "VAHADANE",
                "stain_matrix_target": [
                    [0.65, 0.70, 0.29],
                    [0.07, 0.99, 0.11],
                ],
                "maxC_target": [1.0, 0.8],
            }
        ),
        encoding="utf-8",
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_normalization_manifest(
        master_manifest_path,
        method="VAHADANE",
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )
    monkeypatch.setattr(
        "helpers.training.stain_normalization._load_torch_staintools_builder",
        lambda: _FakeNormalizerBuilder,
    )

    normalizer = build_split_stain_normalizer(
        master_manifest_path,
        [_make_record()],
        runtime_normalization_method="VAHADANE",
        runtime_vahadane_backend="torch_staintools_exact",
    )

    assert normalizer is not None
    assert _FakeNormalizerBuilder.last_method == "vahadane"


def test_build_split_stain_normalizer_reuses_persistent_source_matrix_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "vahadane_stats.json"
    state_path.write_text(
        json.dumps(
            {
                "method": "VAHADANE",
                "stain_matrix_target": [
                    [0.65, 0.70, 0.29],
                    [0.07, 0.99, 0.11],
                ],
                "maxC_target": [1.0, 0.8],
            }
        ),
        encoding="utf-8",
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_normalization_manifest(
        master_manifest_path,
        method="VAHADANE",
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )
    _FakeCachingNormalizerBuilder.instances = []
    monkeypatch.setattr(
        "helpers.training.stain_normalization._load_torch_staintools_builder",
        lambda: _FakeCachingNormalizerBuilder,
    )
    cache_path = tmp_path / "runtime_stain_matrix_cache.sqlite"
    image = np.full((4, 4, 3), 10, dtype=np.uint8)

    first_normalizer = build_split_stain_normalizer(
        master_manifest_path,
        [_make_record()],
        runtime_normalization_method="VAHADANE",
        runtime_vahadane_backend="torch_staintools_exact",
        source_matrix_cache_path=cache_path,
    )
    assert first_normalizer is not None
    first_normalizer.normalize_image(image, cache_key="patch_001.png")

    second_normalizer = build_split_stain_normalizer(
        master_manifest_path,
        [_make_record()],
        runtime_normalization_method="VAHADANE",
        runtime_vahadane_backend="torch_staintools_exact",
        source_matrix_cache_path=cache_path,
    )
    assert second_normalizer is not None
    second_normalizer.normalize_image(image, cache_key="patch_001.png")

    source_fit_calls = [
        module.source_fit_calls for module in _FakeCachingNormalizerBuilder.instances
    ]
    assert source_fit_calls == [1, 0]


def test_build_split_stain_normalizer_resolves_all_supported_methods_from_one_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_payloads = {
        "REINHARD": {
            "method": "REINHARD",
            "target_means": [0.1, 0.2, 0.3],
            "target_stds": [0.4, 0.5, 0.6],
        },
        "RUIFROK": {
            "method": "RUIFROK",
            "stain_matrix_target": [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]],
            "maxC_target": [1.0, 0.8],
        },
        "MACENKO": {
            "method": "MACENKO",
            "stain_matrix_target": [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]],
            "maxC_target": [1.0, 0.8],
        },
        "VAHADANE": {
            "method": "VAHADANE",
            "stain_matrix_target": [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]],
            "maxC_target": [1.0, 0.8],
        },
    }
    artifact_paths: list[tuple[str, Path]] = []
    for method, payload in artifact_payloads.items():
        state_path = tmp_path / f"{method.lower()}_stats.json"
        state_path.write_text(json.dumps(payload), encoding="utf-8")
        artifact_paths.append((method, state_path))

    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_shared_bundle_normalization_manifest(
        master_manifest_path,
        artifacts=artifact_paths,
        stage4_split_bundle_id=11,
    )
    monkeypatch.setattr(
        "helpers.training.stain_normalization._load_torch_staintools_builder",
        lambda: _FakeNormalizerBuilder,
    )

    for method in ("REINHARD", "RUIFROK", "MACENKO", "VAHADANE"):
        normalizer = build_split_stain_normalizer(
            master_manifest_path,
            [_make_record(stage4_split_bundle_id=11)],
            runtime_normalization_method=method,
        )

        assert normalizer is not None
        output = normalizer.normalize_image(
            np.full((4, 4, 3), 10, dtype=np.uint8),
            cache_key=f"{method}.png",
        )
        assert output.shape == (4, 4, 3)
        assert output.dtype == np.uint8


def test_build_split_stain_normalizer_uses_ruifrok_source_matrix_from_state(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "normalization_stats.json"
    state_path.write_text(
        json.dumps(
            {
                "method": "RUIFROK",
                "stain_matrix_source": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                "stain_matrix_target": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                "maxC_target": [1.0, 1.0],
            }
        ),
        encoding="utf-8",
    )
    master_manifest_path = tmp_path / "master_manifest.sqlite"
    _write_normalization_manifest(
        master_manifest_path,
        method="RUIFROK",
        state_path=state_path,
        state_sha256=hash_file_sha256(state_path),
    )

    normalizer = build_split_stain_normalizer(
        master_manifest_path,
        [_make_record()],
        runtime_normalization_method="RUIFROK",
        device="cpu",
    )

    assert normalizer is not None
    module = cast(torch.nn.Module, cast(Any, normalizer).module)
    source_matrix = cast(torch.Tensor, module.stain_matrix_source)
    assert source_matrix.device.type == "cpu"
    assert torch.equal(
        source_matrix,
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32),
    )
