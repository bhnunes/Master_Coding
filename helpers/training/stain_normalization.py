from __future__ import annotations

import json
import sqlite3
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from helpers.extraction.manifest_paths import resolve_manifest_path_ref
from helpers.provenance import hash_file_sha256
from helpers.training.master_manifest_queries import CanonicalRowRecord

SUPPORTED_RUNTIME_NORMALIZATION_METHODS = frozenset(
    {"none", "reinhard", "ruifrok", "macenko", "vahadane"}
)
_RUIFROK_HE_STAIN_MATRIX = torch.tensor(
    [
        [0.644211, 0.716556, 0.266844],
        [0.092789, 0.954111, 0.283111],
    ],
    dtype=torch.float32,
)
_EPSILON = 1e-6
MATRIX_TENSOR_NDIM = 2


class ImageStainNormalizer(Protocol):
    """Normalize one RGB patch before augmentation."""

    def normalize_image(
        self,
        image: npt.NDArray[np.uint8],
        *,
        cache_key: Hashable | None = None,
    ) -> npt.NDArray[np.uint8]: ...


@dataclass(frozen=True)
class NormalizationArtifactRecord:
    """One Stage 5 normalization artifact resolved from SQLite."""

    normalization_artifact_id: int
    method: str
    state_path: Path
    state_sha256: str
    fit_scope: str


def normalize_runtime_method_name(method: str | None) -> str:
    """Map persisted normalization names onto runtime method keys."""

    if method is None:
        return "none"
    normalized = method.strip().lower()
    if normalized == "not_normalized":
        return "none"
    if normalized not in SUPPORTED_RUNTIME_NORMALIZATION_METHODS:
        raise ValueError(f"Unsupported runtime normalization method: {method}")
    return normalized


def build_split_stain_normalizer(
    master_manifest_path: Path,
    records: Sequence[CanonicalRowRecord],
    *,
    device: torch.device | str | None = None,
) -> ImageStainNormalizer | None:
    """Build one shared split-level stain normalizer from Stage 5 metadata."""

    method, normalization_artifact_id = _resolve_split_normalization_metadata(records)
    if method == "none":
        return None
    if normalization_artifact_id is None:
        raise ValueError(
            "Normalized runtime rows are missing normalization_artifact_id in "
            "master_manifest.sqlite."
        )

    artifact = _load_normalization_artifact(master_manifest_path, normalization_artifact_id)
    artifact_method = normalize_runtime_method_name(artifact.method)
    if artifact_method != method:
        raise ValueError(
            "Normalization artifact method does not match split metadata: "
            f"rows={method}, artifact={artifact_method}"
        )
    state = _load_normalization_state_json(artifact)
    runtime_device = torch.device(device) if device is not None else torch.device("cpu")
    return _build_runtime_normalizer(method=method, state=state, device=runtime_device)


def _resolve_split_normalization_metadata(
    records: Sequence[CanonicalRowRecord],
) -> tuple[str, int | None]:
    method_and_artifact_pairs = {
        (
            normalize_runtime_method_name(record.normalization_method),
            record.normalization_artifact_id,
        )
        for record in records
    }
    if not method_and_artifact_pairs:
        return "none", None
    if len(method_and_artifact_pairs) != 1:
        raise ValueError(
            "Runtime split rows do not agree on normalization metadata: "
            f"{sorted(method_and_artifact_pairs)}"
        )
    method, normalization_artifact_id = next(iter(method_and_artifact_pairs))
    if method == "none" and normalization_artifact_id is not None:
        raise ValueError(
            "Non-normalized runtime rows must not reference a normalization artifact id."
        )
    return method, normalization_artifact_id


def _load_normalization_artifact(
    master_manifest_path: Path,
    normalization_artifact_id: int,
) -> NormalizationArtifactRecord:
    with sqlite3.connect(master_manifest_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT normalization_artifact_id, method, state_path, state_sha256, fit_scope
            FROM normalization_artifacts
            WHERE normalization_artifact_id = ?
            """,
            (normalization_artifact_id,),
        ).fetchone()
    if row is None:
        raise ValueError(
            "Normalization artifact id was referenced by runtime rows but is missing from "
            f"master_manifest.sqlite: {normalization_artifact_id}"
        )
    return NormalizationArtifactRecord(
        normalization_artifact_id=int(row["normalization_artifact_id"]),
        method=str(row["method"]),
        state_path=resolve_manifest_path_ref(
            str(row["state_path"]),
            source_root=master_manifest_path.parent,
            manifest_path=master_manifest_path,
        ),
        state_sha256=str(row["state_sha256"]),
        fit_scope=str(row["fit_scope"]),
    )


def _load_normalization_state_json(artifact: NormalizationArtifactRecord) -> dict[str, Any]:
    if not artifact.state_path.is_file():
        raise FileNotFoundError(
            "Normalization state file is missing for artifact "
            f"{artifact.normalization_artifact_id}: "
            f"{artifact.state_path}"
        )
    observed_sha256 = hash_file_sha256(artifact.state_path)
    if observed_sha256 != artifact.state_sha256:
        raise ValueError(
            "Normalization state hash mismatch for artifact "
            f"{artifact.normalization_artifact_id}: expected {artifact.state_sha256}, "
            f"observed {observed_sha256}"
        )
    return cast(dict[str, Any], json.loads(artifact.state_path.read_text(encoding="utf-8")))


def _build_runtime_normalizer(
    *,
    method: str,
    state: dict[str, Any],
    device: torch.device,
) -> ImageStainNormalizer:
    if method == "ruifrok":
        stain_matrix_source = _coerce_optional_tensor(
            state,
            "stain_matrix_source",
            device=device,
        )
        if stain_matrix_source is None:
            stain_matrix_source = _ensure_batch_dimension(
                _RUIFROK_HE_STAIN_MATRIX.to(device=device), 3
            )
        return _TorchModuleImageNormalizer(
            _FixedMatrixDeconvolutionNormalizer(
                stain_matrix_source=stain_matrix_source,
                stain_matrix_target=_coerce_tensor(state, "stain_matrix_target", device=device),
                max_c_target=_coerce_tensor(state, "maxC_target", device=device),
            ).to(device)
        )

    builder = _load_torch_staintools_builder()
    runtime_module = builder.build(
        method,
        concentration_solver="qr",
        use_cache=True,
        device=device,
    )
    _load_runtime_state(runtime_module, method=method, state=state, device=device)
    runtime_module.eval()
    return _TorchModuleImageNormalizer(runtime_module)


def _load_torch_staintools_builder() -> Any:
    try:
        from torch_staintools.normalizer import NormalizerBuilder
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Runtime stain normalization requires the `torch-staintools` package. "
            "Install project dependencies with `uv sync --python 3.12`."
        ) from error
    return NormalizerBuilder


def _load_runtime_state(
    runtime_module: Any,
    *,
    method: str,
    state: dict[str, Any],
    device: torch.device,
) -> None:
    if method == "reinhard":
        _register_or_replace_buffer(
            runtime_module,
            "target_means",
            _coerce_tensor(state, "target_means", device=device),
        )
        _register_or_replace_buffer(
            runtime_module,
            "target_stds",
            _coerce_tensor(state, "target_stds", device=device),
        )
        return

    _register_or_replace_buffer(
        runtime_module,
        "stain_matrix_target",
        _coerce_tensor(state, "stain_matrix_target", device=device),
    )
    _register_or_replace_buffer(
        runtime_module,
        "maxC_target",
        _coerce_tensor(state, "maxC_target", device=device),
    )


def _register_or_replace_buffer(module: Any, name: str, value: torch.Tensor) -> None:
    buffer = value.detach().clone()
    if hasattr(module, name):
        setattr(module, name, buffer)
        return
    register_buffer = getattr(module, "register_buffer", None)
    if register_buffer is None:
        raise TypeError(f"Runtime normalizer backend cannot register buffer: {type(module)!r}")
    register_buffer(name, buffer)


def _coerce_tensor(
    state: dict[str, Any],
    key: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    if key not in state:
        raise ValueError(f"Normalization state is missing required key: {key}")
    tensor = torch.as_tensor(state[key], dtype=torch.float32, device=device)

    def _expand_if_needed(expected_ndim: int, *, add_batch_dim: bool = False) -> torch.Tensor:
        if tensor.ndim == 1:
            return tensor.view(1, -1, 1, 1)
        if tensor.ndim != expected_ndim:
            return tensor
        if add_batch_dim:
            return tensor.unsqueeze(0)
        return tensor.unsqueeze(-1).unsqueeze(-1)

    if key in {"target_means", "target_stds"}:
        return _expand_if_needed(MATRIX_TENSOR_NDIM)
    if key == "stain_matrix_target":
        return _expand_if_needed(MATRIX_TENSOR_NDIM, add_batch_dim=True)
    if key == "maxC_target":
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
    return tensor


def _coerce_optional_tensor(
    state: dict[str, Any],
    key: str,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if key not in state or state[key] is None:
        return None
    return _coerce_tensor(state, key, device=device)


class _TorchModuleImageNormalizer:
    def __init__(self, module: nn.Module) -> None:
        self.module = module.eval()

    @torch.inference_mode()
    def normalize_image(
        self,
        image: npt.NDArray[np.uint8],
        *,
        cache_key: Hashable | None = None,
    ) -> npt.NDArray[np.uint8]:
        image_tensor = _numpy_image_to_tensor(image, device=_module_device(self.module))
        normalized_tensor = _forward_module(self.module, image_tensor, cache_key=cache_key)
        return _tensor_to_numpy_image(normalized_tensor)


class _FixedMatrixDeconvolutionNormalizer(nn.Module):
    """Macenko/Vahadane-style runtime normalizer with a fixed source stain matrix."""

    def __init__(
        self,
        *,
        stain_matrix_source: torch.Tensor,
        stain_matrix_target: torch.Tensor,
        max_c_target: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("stain_matrix_source", _ensure_batch_dimension(stain_matrix_source, 3))
        self.register_buffer("stain_matrix_target", _ensure_batch_dimension(stain_matrix_target, 3))
        self.register_buffer("maxC_target", _ensure_batch_dimension(max_c_target, 2))

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        cache_keys: list[Hashable] | None = None,
    ) -> torch.Tensor:
        del cache_keys
        stain_matrix_source = cast(torch.Tensor, self.stain_matrix_source)
        max_c_target = cast(torch.Tensor, self.maxC_target)
        stain_matrix_target = cast(torch.Tensor, self.stain_matrix_target)
        image_od = _rgb_to_od(x)
        flattened_od = image_od.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 3)
        source_matrix = stain_matrix_source.expand(x.shape[0], -1, -1)
        source_concentration = _solve_concentration(flattened_od, source_matrix)
        max_c = torch.quantile(source_concentration, q=0.99, dim=1).clamp_min(_EPSILON)
        scale = (max_c_target / max_c).unsqueeze(1)
        normalized_concentration = source_concentration * scale
        target_matrix = stain_matrix_target.expand(x.shape[0], -1, -1)
        reconstructed_od = normalized_concentration @ target_matrix
        reconstructed_rgb = torch.exp(-reconstructed_od).reshape(
            x.shape[0], x.shape[2], x.shape[3], 3
        )
        return reconstructed_rgb.permute(0, 3, 1, 2).clamp_(0.0, 1.0)


def _ensure_batch_dimension(tensor: torch.Tensor, trailing_dims: int) -> torch.Tensor:
    if tensor.ndim == trailing_dims - 1:
        return tensor.unsqueeze(0)
    return tensor


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _forward_module(
    module: nn.Module,
    image_tensor: torch.Tensor,
    *,
    cache_key: Hashable | None,
) -> torch.Tensor:
    if cache_key is not None:
        try:
            return cast(torch.Tensor, module(image_tensor, cache_keys=[cache_key]))
        except TypeError:
            pass
    return cast(torch.Tensor, module(image_tensor))


def _numpy_image_to_tensor(
    image: npt.NDArray[np.uint8],
    *,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(image, dtype=np.uint8))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def _tensor_to_numpy_image(image_tensor: torch.Tensor) -> npt.NDArray[np.uint8]:
    image = (
        image_tensor.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return cast(npt.NDArray[np.uint8], image)


def _rgb_to_od(x: torch.Tensor) -> torch.Tensor:
    return -torch.log(x.clamp_min(_EPSILON))


def _solve_concentration(flattened_od: torch.Tensor, stain_matrix: torch.Tensor) -> torch.Tensor:
    solved = cast(
        torch.Tensor,
        torch.linalg.lstsq(stain_matrix.transpose(1, 2), flattened_od.transpose(1, 2)).solution,
    )
    return solved.transpose(1, 2).clamp_min(0.0)
