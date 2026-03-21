from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn

from helpers.ensemble_optimizer import optimization
from helpers.ensemble_optimizer.config import EnsembleOptimizerConfig
from helpers.ensemble_optimizer.optimization import (
    _calibrate_decision_threshold,
    _compute_positive_patients,
    _normalize_weights,
    _weighted_ensemble_from_u16_cache,
    cache_predictions_sequential,
    compute_patient_auprc_in_roi,
    generate_roi_batch,
    get_stream_type,
    predict_with_tta_batched,
    run_two_stream_optimization,
)


class _ConstantBinaryModel(nn.Module):
    def __init__(self, value: float, arch_name: str = "SWIN") -> None:
        super().__init__()
        self.value = value
        self.arch_name = arch_name

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (images.shape[0], 1, images.shape[2], images.shape[3]),
            self.value,
            dtype=torch.float32,
            device=images.device,
        )


class _TupleTwoClassModel(nn.Module):
    def __init__(self, arch_name: str = "FPN") -> None:
        super().__init__()
        self.arch_name = arch_name

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.zeros(
            (images.shape[0], 2, images.shape[2], images.shape[3]),
            dtype=torch.float32,
            device=images.device,
        )
        output[:, 1] = 1.0
        aux = torch.zeros((), dtype=torch.float32, device=images.device)
        return output, aux


class _ListLoader:
    def __init__(self, dataset_len: int, batches: list[Any]) -> None:
        self.dataset = list(range(dataset_len))
        self._batches = batches

    def __iter__(self) -> Iterator[Any]:
        return iter(self._batches)


@pytest.fixture
def optimizer_config(tmp_path: Path) -> EnsembleOptimizerConfig:
    return EnsembleOptimizerConfig(
        hdf5_drive_dir=tmp_path / "dataset",
        metadata_dir=tmp_path / "metadata",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "local",
        pred_cache_dir=tmp_path / "cache",
        stage_input_locally=False,
        overwrite_output=True,
        seed=24,
        batch_size=2,
        workers=0,
        top_models=2,
        sort_metric="best_val_auprc_pixel_score",
        val_calibration_frac=0.25,
        val_holdout_frac=0.5,
        semantic_architectures=("SWIN",),
        spatial_architectures=("FPN",),
        roi_context_scale=1,
        roi_max_median=1.0,
        roi_empty_max=1.0,
        roi_min_pos_recall=0.0,
        spill_penalty_lambda=0.1,
        spatial_patient_policy="positive_only",
        num_trials_semantic=1,
        num_trials_spatial=1,
    )


def _write_prediction_cache(
    cache_dir: Path,
    *,
    predictions: list[npt.NDArray[np.uint16]],
    truths: npt.NDArray[np.uint8],
    patient_ids: list[str],
) -> tuple[list[Path], Path, Path, int, int, int, np.memmap[Any, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    height = int(truths.shape[1])
    width = int(truths.shape[2])
    truth_path = cache_dir / "trues.dat"
    truth_memmap = np.memmap(truth_path, dtype="uint8", mode="w+", shape=truths.shape)
    truth_memmap[:] = truths
    truth_memmap.flush()
    prediction_paths: list[Path] = []
    for index, values in enumerate(predictions):
        path = cache_dir / f"pred_{index}.dat"
        memmap = np.memmap(path, dtype="uint16", mode="w+", shape=values.shape)
        memmap[:] = values
        memmap.flush()
        prediction_paths.append(path)
    pids_path = cache_dir / "pids.json"
    pids_path.write_text(json.dumps(patient_ids), encoding="utf-8")
    return prediction_paths, truth_path, pids_path, truths.shape[0], height, width, truth_memmap


def test_get_stream_type_returns_semantic_spatial_or_default_spatial(
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    assert get_stream_type("swin", optimizer_config) == "semantic"
    assert get_stream_type("fpn", optimizer_config) == "spatial"
    assert get_stream_type("unknown", optimizer_config) == "spatial"


def test_generate_roi_batch_downscales_thresholds_and_restores_shape() -> None:
    probabilities = torch.tensor(
        [[[0.9, 0.9, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]],
        dtype=torch.float32,
    )

    roi = generate_roi_batch(probabilities, context_scale=2, threshold=0.5)

    assert tuple(roi.shape) == (1, 4, 4)
    assert torch.equal(roi[0, :2, :2], torch.ones((2, 2)))
    assert torch.equal(roi[0, 2:, 2:], torch.zeros((2, 2)))


def test_compute_patient_auprc_in_roi_returns_zero_for_empty_roi_or_no_positives() -> None:
    assert (
        compute_patient_auprc_in_roi(np.array([1, 0]), np.array([0.8, 0.2]), np.array([0, 0]))
        == 0.0
    )
    assert (
        compute_patient_auprc_in_roi(np.array([0, 0]), np.array([0.8, 0.2]), np.array([1, 1]))
        == 0.0
    )


def test_compute_patient_auprc_in_roi_returns_zero_when_metric_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        optimization,
        "average_precision_score",
        lambda y_true, y_pred: (_ for _ in ()).throw(ValueError("boom")),
    )

    result = compute_patient_auprc_in_roi(np.array([1, 0]), np.array([0.8, 0.2]), np.array([1, 1]))

    assert result == 0.0


def test_weighted_ensemble_from_u16_cache_skips_near_zero_weights() -> None:
    arrays = [
        np.full((1, 2, 2), 65535, dtype=np.uint16),
        np.full((1, 2, 2), 32768, dtype=np.uint16),
    ]

    result = _weighted_ensemble_from_u16_cache(
        cast(list[npt.NDArray[np.generic]], arrays), [1.0, 0.0]
    )

    assert np.allclose(result, np.ones((1, 2, 2), dtype=np.float32))


def test_weighted_ensemble_from_u16_cache_returns_zeros_when_all_weights_are_zero() -> None:
    arrays = [np.full((1, 2, 2), 65535, dtype=np.uint16)]

    result = _weighted_ensemble_from_u16_cache(cast(list[npt.NDArray[np.generic]], arrays), [0.0])

    assert np.array_equal(result, np.zeros((1, 2, 2), dtype=np.float32))


def test_normalize_weights_returns_probabilities_summing_to_one() -> None:
    normalized = _normalize_weights([2.0, 1.0, 1.0])

    assert sum(normalized) == pytest.approx(1.0)
    assert normalized[0] == pytest.approx(0.5)


def test_calibrate_decision_threshold_maximizes_patient_mcc(tmp_path: Path) -> None:
    truths = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    cache_payload = _write_prediction_cache(
        tmp_path / "pred-cache",
        predictions=[
            np.array(
                [
                    np.full((2, 2), 65535, dtype=np.uint16),
                    np.zeros((2, 2), dtype=np.uint16),
                ]
            ),
            np.array(
                [
                    np.array([[39321, 13107], [13107, 13107]], dtype=np.uint16),
                    np.array([[26214, 26214], [26214, 26214]], dtype=np.uint16),
                ]
            ),
        ],
        truths=truths,
        patient_ids=["p1", "p2"],
    )
    prediction_paths, _truth_path, _pids_path, total_samples, height, width, truth_memmap = (
        cache_payload
    )
    prediction_memmaps = [
        np.memmap(path, dtype="uint16", mode="r", shape=(total_samples, height, width))
        for path in prediction_paths
    ]

    threshold, metrics = _calibrate_decision_threshold(
        patient_ids=["p1", "p2"],
        local_map={"p1": slice(0, 1), "p2": slice(1, 2)},
        global_indices=np.array([0, 1]),
        truth_memmap=truth_memmap,
        prediction_memmaps=prediction_memmaps,
        semantic_indices=[0],
        semantic_weights=[1.0],
        spatial_indices=[1],
        spatial_weights=[1.0],
        roi_context_scale=1,
        roi_threshold=0.5,
    )

    assert threshold == pytest.approx(0.2)
    assert metrics["Calibration_best_mcc"] == pytest.approx(0.5)
    assert metrics["Calibration_n_positive_patients"] == 1
    assert metrics["Calibration_n_negative_patients"] == 1


def test_compute_positive_patients_flags_any_patient_with_positive_pixel(tmp_path: Path) -> None:
    truth_path = tmp_path / "truth.dat"
    truth = np.memmap(truth_path, dtype="uint8", mode="w+", shape=(3, 2, 2))
    truth[:] = 0
    truth[1, 0, 0] = 1
    truth.flush()

    positive = _compute_positive_patients({"p1": [0], "p2": [1, 2]}, truth)

    assert positive == {"p2"}


def test_optimizer_predict_with_tta_batched_handles_binary_and_multiclass_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(optimization, "autocast_ctx", lambda x, amp_dtype: contextlib.nullcontext())
    images = torch.zeros((1, 3, 2, 2), dtype=torch.float32)

    binary = predict_with_tta_batched(_ConstantBinaryModel(0.0), images, "SWIN")
    multiclass = predict_with_tta_batched(_TupleTwoClassModel(), images, "FPN")
    expected_prob = float(torch.softmax(torch.tensor([0.0, 1.0]), dim=0)[1])

    assert torch.allclose(binary, torch.full((1, 2, 2), 0.5))
    assert torch.allclose(
        multiclass,
        torch.full((1, 2, 2), expected_prob),
    )


def test_cache_predictions_sequential_writes_memmaps_and_patient_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = (
        torch.zeros((2, 3, 2, 2), dtype=torch.uint8),
        torch.tensor(
            [
                [[[1, 1], [1, 1]], [[0, 1], [1, 0]]],
                [[[1, 1], [1, 1]], [[1, 0], [0, 1]]],
            ],
            dtype=torch.uint8,
        ),
        ["p1", "p2"],
    )
    dataloader = _ListLoader(dataset_len=2, batches=[batch])
    monkeypatch.setattr(
        optimization,
        "predict_with_tta_batched",
        lambda model, images, architecture: torch.tensor(
            [[[0.0, 0.5], [1.0, float("nan")]], [[float("inf"), 0.25], [0.75, float("-inf")]]],
            dtype=torch.float32,
        ),
    )
    monkeypatch.setattr(optimization, "clear_gpu", lambda: None)

    prediction_paths, trues_path, pids_path, total_samples, height, width, truth_memmap = (
        cache_predictions_sequential(
            [_ConstantBinaryModel(0.0)],
            cast(Any, dataloader),
            device=torch.device("cpu"),
            cache_dir=tmp_path / "cache",
        )
    )

    prediction_memmap = np.memmap(
        prediction_paths[0],
        dtype="uint16",
        mode="r",
        shape=(total_samples, height, width),
    )
    assert trues_path.exists()
    assert truth_memmap.shape == (2, 2, 2)
    assert prediction_memmap[0, 1, 0] == 65535
    assert prediction_memmap[0, 1, 1] == 0
    assert prediction_memmap[1, 0, 0] == 65535
    assert prediction_memmap[1, 1, 1] == 0
    assert json.loads(pids_path.read_text(encoding="utf-8")) == ["p1", "p2"]


def test_cache_predictions_sequential_raises_when_valid_samples_do_not_match_dataset_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = (
        torch.zeros((2, 3, 2, 2), dtype=torch.uint8),
        torch.zeros((2, 2, 2, 2), dtype=torch.uint8),
        ["p1", "p2"],
    )
    dataloader = _ListLoader(dataset_len=3, batches=[batch, None])
    monkeypatch.setattr(
        optimization,
        "predict_with_tta_batched",
        lambda model, images, architecture: torch.zeros((2, 2, 2), dtype=torch.float32),
    )
    monkeypatch.setattr(optimization, "clear_gpu", lambda: None)

    with pytest.raises(RuntimeError, match="Cached 2 samples but expected 3"):
        cache_predictions_sequential(
            [_ConstantBinaryModel(0.0)],
            cast(Any, dataloader),
            device=torch.device("cpu"),
            cache_dir=tmp_path / "cache",
        )


def test_run_two_stream_optimization_raises_when_no_semantic_models_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    cache_payload = _write_prediction_cache(
        tmp_path / "pred-cache",
        predictions=[np.full((2, 2, 2), 1000, dtype=np.uint16)],
        truths=np.zeros((2, 2, 2), dtype=np.uint8),
        patient_ids=["p1", "p2"],
    )
    monkeypatch.setattr(
        optimization, "cache_predictions_sequential", lambda *args, **kwargs: cache_payload
    )

    with pytest.raises(ValueError, match="No semantic models found"):
        run_two_stream_optimization(
            optimizer_config,
            [_ConstantBinaryModel(0.0, arch_name="FPN")],
            cast(Any, _ListLoader(2, [])),
            device=torch.device("cpu"),
        )


def test_run_two_stream_optimization_falls_back_to_semantic_models_for_spatial_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    cache_payload = _write_prediction_cache(
        tmp_path / "pred-cache",
        predictions=[np.full((2, 2, 2), 32768, dtype=np.uint16)],
        truths=np.zeros((2, 2, 2), dtype=np.uint8),
        patient_ids=["p1", "p2"],
    )
    config = replace(optimizer_config, spatial_architectures=())

    monkeypatch.setattr(
        optimization, "cache_predictions_sequential", lambda *args, **kwargs: cache_payload
    )
    monkeypatch.setattr(
        optimization,
        "build_holdout_split",
        lambda patient_ids, positive_patients, calibration_frac, holdout_frac, seed: (
            SimpleNamespace(
                holdout_patients={"p2"}, calibration_patients=set(), optimization_patients={"p1"}
            )
        ),
    )
    monkeypatch.setattr(
        optimization,
        "build_indices_and_local_map",
        lambda selected_patients, patient_map: (
            np.array([0]) if selected_patients == {"p1"} else np.array([1]),
            {"p1": slice(0, 1)} if selected_patients == {"p1"} else {"p2": slice(0, 1)},
            ["p1"] if selected_patients == {"p1"} else ["p2"],
        ),
    )

    class _FakeStudy:
        def __init__(self, best_params: dict[str, float]) -> None:
            self.best_params = best_params

        def optimize(self, objective: Any, n_trials: int) -> None:
            del objective, n_trials

    studies = iter([_FakeStudy({"w_sem_0": 1.0, "roi_thresh": 0.5}), _FakeStudy({"w_spa_0": 1.0})])
    optuna_module = optimization.optuna  # type: ignore[attr-defined]
    monkeypatch.setattr(optuna_module, "create_study", lambda direction, sampler: next(studies))
    monkeypatch.setattr(
        optimization,
        "generate_roi_batch",
        lambda probabilities, context_scale, threshold: torch.ones_like(probabilities),
    )

    result = run_two_stream_optimization(
        config,
        [_ConstantBinaryModel(0.0, arch_name="SWIN")],
        cast(Any, _ListLoader(2, [])),
        device=torch.device("cpu"),
    )

    assert result.semantic_indices == [0]
    assert result.spatial_indices == [0]
    assert result.holdout_metrics["N_eval_patients"] == 0
    assert result.holdout_metrics["Spatial_patient_policy"] == "positive_only"


def test_run_two_stream_optimization_returns_holdout_metrics_for_positive_only_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    truths = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    predictions = [
        np.full((2, 2, 2), 32768, dtype=np.uint16),
        np.full((2, 2, 2), 32768, dtype=np.uint16),
    ]
    cache_payload = _write_prediction_cache(
        tmp_path / "pred-cache",
        predictions=predictions,
        truths=truths,
        patient_ids=["p1", "p2"],
    )
    monkeypatch.setattr(
        optimization, "cache_predictions_sequential", lambda *args, **kwargs: cache_payload
    )
    monkeypatch.setattr(
        optimization,
        "build_holdout_split",
        lambda patient_ids, positive_patients, calibration_frac, holdout_frac, seed: (
            SimpleNamespace(
                holdout_patients={"p1"}, calibration_patients=set(), optimization_patients={"p2"}
            )
        ),
    )
    monkeypatch.setattr(
        optimization,
        "build_indices_and_local_map",
        lambda selected_patients, patient_map: (
            np.array([1])
            if selected_patients == {"p2"}
            else (np.array([], dtype=np.int64) if not selected_patients else np.array([0])),
            {"p2": slice(0, 1)}
            if selected_patients == {"p2"}
            else ({} if not selected_patients else {"p1": slice(0, 1)}),
            ["p2"] if selected_patients == {"p2"} else ([] if not selected_patients else ["p1"]),
        ),
    )

    class _FakeStudy:
        def __init__(self, best_params: dict[str, float]) -> None:
            self.best_params = best_params

        def optimize(self, objective: Any, n_trials: int) -> None:
            del objective, n_trials

    studies = iter([_FakeStudy({"w_sem_0": 1.0, "roi_thresh": 0.5}), _FakeStudy({"w_spa_0": 1.0})])
    optuna_module = optimization.optuna  # type: ignore[attr-defined]
    monkeypatch.setattr(optuna_module, "create_study", lambda direction, sampler: next(studies))
    monkeypatch.setattr(
        optimization,
        "generate_roi_batch",
        lambda probabilities, context_scale, threshold: torch.ones_like(probabilities),
    )
    monkeypatch.setattr(
        optimization, "compute_patient_auprc_in_roi", lambda y_true, y_pred, roi_mask: 0.75
    )

    result = run_two_stream_optimization(
        optimizer_config,
        [_ConstantBinaryModel(0.0, arch_name="SWIN"), _TupleTwoClassModel(arch_name="FPN")],
        cast(Any, _ListLoader(2, [])),
        device=torch.device("cpu"),
    )

    assert result.semantic_indices == [0]
    assert result.spatial_indices == [1]
    assert result.roi_threshold == 0.5
    assert result.decision_threshold == 0.5
    assert result.calibration_metrics["Calibration_metric"] == "Patient_MCC"
    assert result.holdout_metrics["Macro_AUPRC_in_ROI"] == pytest.approx(0.75)
    assert result.holdout_metrics["Macro_Spill"] == pytest.approx(0.0)
    assert result.holdout_metrics["N_eval_patients"] == 1
    assert result.holdout_metrics["N_pos_patients_total"] == 1


def test_run_two_stream_optimization_all_policy_penalizes_negative_false_positives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    truths = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    high_predictions = [
        np.full((4, 2, 2), 65535, dtype=np.uint16),
        np.full((4, 2, 2), 65535, dtype=np.uint16),
    ]
    cache_payload = _write_prediction_cache(
        tmp_path / "pred-cache",
        predictions=high_predictions,
        truths=truths,
        patient_ids=["p1", "p2", "p3", "p4"],
    )
    monkeypatch.setattr(
        optimization, "cache_predictions_sequential", lambda *args, **kwargs: cache_payload
    )
    monkeypatch.setattr(
        optimization,
        "build_holdout_split",
        lambda patient_ids, positive_patients, calibration_frac, holdout_frac, seed: (
            SimpleNamespace(
                holdout_patients={"p3", "p4"},
                calibration_patients=set(),
                optimization_patients={"p1", "p2"},
            )
        ),
    )

    def fake_build_indices_and_local_map(
        selected_patients: set[str],
        patient_map: dict[str, list[int]],
    ) -> tuple[npt.NDArray[np.int64], dict[str, slice], list[str]]:
        del patient_map
        if selected_patients == {"p1", "p2"}:
            return np.array([0, 1]), {"p1": slice(0, 1), "p2": slice(1, 2)}, ["p1", "p2"]
        return np.array([2, 3]), {"p3": slice(0, 1), "p4": slice(1, 2)}, ["p3", "p4"]

    monkeypatch.setattr(
        optimization, "build_indices_and_local_map", fake_build_indices_and_local_map
    )

    class _FakeTrial:
        def suggest_float(self, name: str, low: float, high: float) -> float:
            del low, high
            return 1.0 if name.startswith("w_") else 0.5

    class _FakeStudy:
        def __init__(self, best_params: dict[str, float]) -> None:
            self.best_params = best_params

        def optimize(self, objective: Any, n_trials: int) -> None:
            del n_trials
            objective(_FakeTrial())

    studies = iter([_FakeStudy({"w_sem_0": 1.0, "roi_thresh": 0.5}), _FakeStudy({"w_spa_0": 1.0})])
    optuna_module = optimization.optuna  # type: ignore[attr-defined]
    monkeypatch.setattr(optuna_module, "create_study", lambda direction, sampler: next(studies))
    monkeypatch.setattr(
        optimization,
        "generate_roi_batch",
        lambda probabilities, context_scale, threshold: torch.ones_like(probabilities),
    )
    monkeypatch.setattr(
        optimization, "compute_patient_auprc_in_roi", lambda y_true, y_pred, roi_mask: 0.8
    )

    result = run_two_stream_optimization(
        replace(optimizer_config, spatial_patient_policy="all"),
        [_ConstantBinaryModel(0.0, arch_name="SWIN"), _TupleTwoClassModel(arch_name="FPN")],
        cast(Any, _ListLoader(4, [])),
        device=torch.device("cpu"),
    )

    assert result.holdout_metrics["Spatial_patient_policy"] == "all"
    assert result.holdout_metrics["N_eval_patients"] == 2
    assert result.holdout_metrics["N_eval_negative_patients"] == 1
    assert result.holdout_metrics["Macro_AUPRC_in_ROI_Positive"] == pytest.approx(0.8)
    assert result.holdout_metrics["Macro_Negative_FP"] == pytest.approx(1.0)
    assert result.holdout_metrics["Macro_Spill_All"] == pytest.approx(0.0)
    assert result.holdout_metrics["Objective_Composite"] == pytest.approx(0.7)
