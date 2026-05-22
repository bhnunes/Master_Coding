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
import optuna
import pytest
import torch
from torch import nn

from helpers.ensemble_optimizer import optimization
from helpers.ensemble_optimizer.config import EnsembleOptimizerConfig
from helpers.ensemble_optimizer.optimization import (
    _build_optimization_patient_caches,
    _calibrate_decision_threshold,
    _compute_positive_patients,
    _normalize_weights,
    _optimization_patient_semantic_prediction,
    _optimization_patient_truth,
    _weighted_ensemble_from_stacked_u16,
    _weighted_ensemble_from_u16_cache,
    cache_predictions_sequential,
    compute_patient_auprc_in_roi,
    generate_roi_batch,
    get_stream_type,
    predict_with_tta_batched,
    run_two_stream_optimization,
)
from helpers.ensemble_optimizer.splitting import HoldoutSplit

UINT16_MAX = 65535
DEFAULT_THRESHOLD = 0.5
EVAL_PATIENT_COUNT = 2
LOW_CACHE_VALUE = 1000
MID_CACHE_VALUE = 2000
HIGH_CACHE_VALUE = 3000
TOP_CACHE_VALUE = 4000
RULE6_COMPONENT_AREA_CANDIDATE = 2
V3_NEAR_MICRO_DICE_FLOOR = 0.80


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


class _FixedTrial:
    def __init__(self, params: dict[str, float]) -> None:
        self._params = params

    def suggest_float(self, name: str, low: float, high: float) -> float:
        del low, high
        return self._params[name]


@pytest.fixture
def optimizer_config(tmp_path: Path) -> EnsembleOptimizerConfig:
    return EnsembleOptimizerConfig(
        master_manifest_path=tmp_path / "dataset" / "master_manifest.sqlite",
        metadata_dir=tmp_path / "metadata",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "local",
        pred_cache_dir=tmp_path / "cache",
        stage_input_locally=False,
        overwrite_output=True,
        seed=24,
        batch_size=2,
        workers=0,
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
        min_micro_dice=0.0,
        negative_clean_target=0.0,
        min_macro_precision=0.0,
        min_macro_tpr=0.0,
        min_patient_positive_area_fraction_candidates=(0.0,),
        min_component_area_fraction_patch_candidates=(0.0,),
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


def test_get_stream_type_returns_semantic_spatial_or_none(
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    assert get_stream_type("swin", optimizer_config) == "semantic"
    assert get_stream_type("fpn", optimizer_config) == "spatial"
    assert get_stream_type("unknown", optimizer_config) == "none"


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


def test_weighted_ensemble_from_stacked_u16_combines_model_axis() -> None:
    stacked = np.asarray(
        [
            np.full((1, 2, 2), UINT16_MAX, dtype=np.uint16),
            np.zeros((1, 2, 2), dtype=np.uint16),
        ]
    )

    result = _weighted_ensemble_from_stacked_u16(stacked, [0.75, 0.25])

    assert np.allclose(result, np.full((1, 2, 2), 0.75, dtype=np.float32))


def test_build_optimization_patient_caches_stacks_predictions_by_patient(tmp_path: Path) -> None:
    cache_payload = _write_prediction_cache(
        tmp_path / "pred-cache",
        predictions=[
            np.array(
                [
                    np.full((2, 2), LOW_CACHE_VALUE, dtype=np.uint16),
                    np.full((2, 2), MID_CACHE_VALUE, dtype=np.uint16),
                ]
            ),
            np.array(
                [
                    np.full((2, 2), HIGH_CACHE_VALUE, dtype=np.uint16),
                    np.full((2, 2), TOP_CACHE_VALUE, dtype=np.uint16),
                ]
            ),
        ],
        truths=np.array([np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)]),
        patient_ids=["p1", "p2"],
    )
    (
        prediction_paths,
        _truth_path,
        _pids_path,
        total_samples,
        height,
        width,
        truth_memmap,
    ) = cache_payload
    prediction_memmaps = [
        np.memmap(path, dtype="uint16", mode="r", shape=(total_samples, height, width))
        for path in prediction_paths
    ]
    semantic_cache, spatial_cache, truth_cache, gt_density_by_patient = (
        _build_optimization_patient_caches(
            optimization._OptimizationCacheBuildInput(
                prediction_memmaps=prediction_memmaps,
                truth_memmap=truth_memmap,
                optimization_idx=np.array([0, 1]),
                optimization_local_map={"p1": slice(0, 1), "p2": slice(1, 2)},
                optimization_patients=["p1", "p2"],
                semantic_indices=[0],
                spatial_indices=[1],
                height=height,
                width=width,
                max_cache_bytes=1024 * 1024,
            )
        )
    )

    assert semantic_cache is not None
    assert spatial_cache is not None
    assert truth_cache is not None
    assert gt_density_by_patient is not None
    assert semantic_cache["p1"].shape == (1, 1, 2, 2)
    assert int(semantic_cache["p1"][0, 0, 0, 0]) == LOW_CACHE_VALUE
    assert int(spatial_cache["p2"][0, 0, 0, 0]) == TOP_CACHE_VALUE
    assert np.array_equal(truth_cache["p2"], np.ones((1, 2, 2), dtype=np.uint8))
    assert gt_density_by_patient["p1"] == pytest.approx(0.0)
    assert gt_density_by_patient["p2"] == pytest.approx(1.0)


def test_optimization_patient_helpers_prefer_cached_arrays() -> None:
    prepared = optimization.OptimizationPreparation(
        patient_map={"p1": [0]},
        prediction_memmaps=[],
        truth_memmap=cast(Any, np.zeros((1, 2, 2), dtype=np.uint8)),
        holdout_idx=np.array([], dtype=np.int64),
        holdout_local_map={},
        holdout_patients=[],
        calibration_idx=np.array([], dtype=np.int64),
        calibration_local_map={},
        calibration_patients=[],
        optimization_idx=np.array([0], dtype=np.int64),
        optimization_local_map={"p1": slice(0, 1)},
        optimization_patients=["p1"],
        semantic_indices=[0],
        spatial_indices=[0],
        optimization_truth=np.zeros((1, 2, 2), dtype=np.uint8),
        optimization_semantic_cache={
            "p1": np.asarray([np.full((1, 2, 2), UINT16_MAX, dtype=np.uint16)])
        },
        optimization_spatial_cache={
            "p1": np.asarray([np.full((1, 2, 2), UINT16_MAX, dtype=np.uint16)])
        },
        optimization_truth_cache={"p1": np.ones((1, 2, 2), dtype=np.uint8)},
        optimization_gt_density_by_patient={"p1": 1.0},
        optimization_positive_patients={"p1"},
        height=2,
        width=2,
    )

    semantic_prediction = _optimization_patient_semantic_prediction(
        prepared,
        patient_id="p1",
        weights=[1.0],
    )
    patient_truth = _optimization_patient_truth(prepared, patient_id="p1")

    assert np.allclose(semantic_prediction, np.ones((1, 2, 2), dtype=np.float32))
    assert np.array_equal(patient_truth, np.ones((1, 2, 2), dtype=np.uint8))


def test_prepare_optimization_builds_memory_capped_patient_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, optimizer_config: EnsembleOptimizerConfig
) -> None:
    cache_payload = _write_prediction_cache(
        tmp_path / "pred-cache",
        predictions=[
            np.array(
                [
                    np.full((2, 2), LOW_CACHE_VALUE, dtype=np.uint16),
                    np.full((2, 2), MID_CACHE_VALUE, dtype=np.uint16),
                ]
            ),
            np.array(
                [
                    np.full((2, 2), HIGH_CACHE_VALUE, dtype=np.uint16),
                    np.full((2, 2), TOP_CACHE_VALUE, dtype=np.uint16),
                ]
            ),
        ],
        truths=np.array([np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)]),
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
                holdout_patients={"p2"},
                calibration_patients=set(),
                optimization_patients={"p1"},
            )
        ),
    )
    monkeypatch.setattr(
        optimization,
        "build_indices_and_local_map",
        lambda selected_patients, patient_map: (
            np.array([0])
            if selected_patients == {"p1"}
            else (np.array([], dtype=np.int64) if not selected_patients else np.array([1])),
            {"p1": slice(0, 1)}
            if selected_patients == {"p1"}
            else ({} if not selected_patients else {"p2": slice(0, 1)}),
            ["p1"] if selected_patients == {"p1"} else ([] if not selected_patients else ["p2"]),
        ),
    )
    semantic_model = _ConstantBinaryModel(0.0, arch_name="SWIN")
    spatial_model = _TupleTwoClassModel(arch_name="FPN")

    prepared = optimization._prepare_optimization(
        optimizer_config,
        [semantic_model, spatial_model],
        cast(Any, _ListLoader(2, [])),
        device=torch.device("cpu"),
        predefined_split=None,
    )

    assert prepared.optimization_semantic_cache is not None
    assert prepared.optimization_spatial_cache is not None
    assert prepared.optimization_truth_cache is not None
    assert int(prepared.optimization_semantic_cache["p1"][0, 0, 0, 0]) == LOW_CACHE_VALUE
    assert int(prepared.optimization_spatial_cache["p1"][0, 0, 0, 0]) == HIGH_CACHE_VALUE
    assert np.array_equal(
        prepared.optimization_truth_cache["p1"],
        np.zeros((1, 2, 2), dtype=np.uint8),
    )


def test_prepare_optimization_uses_memmap_fallback_when_cache_cap_is_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, optimizer_config: EnsembleOptimizerConfig
) -> None:
    cache_payload = _write_prediction_cache(
        tmp_path / "pred-cache",
        predictions=[
            np.array(
                [
                    np.full((2, 2), LOW_CACHE_VALUE, dtype=np.uint16),
                    np.full((2, 2), MID_CACHE_VALUE, dtype=np.uint16),
                ]
            ),
            np.array(
                [
                    np.full((2, 2), HIGH_CACHE_VALUE, dtype=np.uint16),
                    np.full((2, 2), TOP_CACHE_VALUE, dtype=np.uint16),
                ]
            ),
        ],
        truths=np.array([np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)]),
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
                holdout_patients={"p2"},
                calibration_patients=set(),
                optimization_patients={"p1"},
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

    prepared = optimization._prepare_optimization(
        replace(optimizer_config, optimization_cache_max_bytes=1),
        [_ConstantBinaryModel(0.0, arch_name="SWIN"), _TupleTwoClassModel(arch_name="FPN")],
        cast(Any, _ListLoader(2, [])),
        device=torch.device("cpu"),
        predefined_split=None,
    )

    assert prepared.optimization_semantic_cache is None
    assert prepared.optimization_spatial_cache is None
    assert prepared.optimization_truth_cache is None
    semantic_prediction = _optimization_patient_semantic_prediction(
        prepared,
        patient_id="p1",
        weights=[1.0],
    )
    assert semantic_prediction[0, 0, 0] == pytest.approx(LOW_CACHE_VALUE / UINT16_MAX)


def test_normalize_weights_returns_probabilities_summing_to_one() -> None:
    normalized = _normalize_weights([2.0, 1.0, 1.0])

    assert sum(normalized) == pytest.approx(1.0)
    assert normalized[0] == pytest.approx(0.5)


def test_calibrate_decision_threshold_selects_higher_rule6_candidate(
    tmp_path: Path,
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    truths = np.array(
        [
            [[1, 1], [1, 1]],
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
                    np.full((2, 2), 65535, dtype=np.uint16),
                ]
            ),
            np.array(
                [
                    np.full((2, 2), 52428, dtype=np.uint16),
                    np.array([[52428, 0], [0, 0]], dtype=np.uint16),
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

    result = _calibrate_decision_threshold(
        optimization.ThresholdCalibrationConfig(
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
            optimizer_config=replace(
                optimizer_config,
                decision_threshold_min=0.5,
                decision_threshold_max=0.5,
                decision_threshold_step=0.1,
                min_component_area_px_candidates=(0, RULE6_COMPONENT_AREA_CANDIDATE),
                min_patient_positive_patches_candidates=(1,),
            ),
        )
    )

    assert result.decision_threshold == pytest.approx(0.5)
    assert result.postprocessing_config.min_component_area_px == RULE6_COMPONENT_AREA_CANDIDATE
    assert result.postprocessing_config.min_patient_positive_patches == 1
    assert result.metrics["Calibration_metric"] == "Macro_Rule6_Dice"
    assert result.metrics["Calibration_macro_rule6"] == pytest.approx(1.0)
    assert result.metrics["Calibration_positive_tpr"] == pytest.approx(1.0)
    assert result.metrics["Calibration_n_positive_patients"] == 1
    assert result.metrics["Calibration_n_negative_patients"] == 1
    assert result.summary["selected"]["min_component_area_px"] == RULE6_COMPONENT_AREA_CANDIDATE
    assert result.summary["selected"]["positive_tpr"] == pytest.approx(1.0)


def test_calibrate_decision_threshold_rejects_positive_dice_drop(
    tmp_path: Path,
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    truths = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    spatial_predictions = np.array(
        [
            [[52428, 0], [0, 0]],
            [[52428, 0], [0, 0]],
            [[52428, 0], [0, 0]],
            [[52428, 0], [0, 0]],
            [[52428, 0], [0, 0]],
        ],
        dtype=np.uint16,
    )
    cache_payload = _write_prediction_cache(
        tmp_path / "pred-cache",
        predictions=[
            np.full((5, 2, 2), UINT16_MAX, dtype=np.uint16),
            spatial_predictions,
        ],
        truths=truths,
        patient_ids=["p1", "p2", "p3", "p4", "p5"],
    )
    prediction_paths, _truth_path, _pids_path, total_samples, height, width, truth_memmap = (
        cache_payload
    )
    prediction_memmaps = [
        np.memmap(path, dtype="uint16", mode="r", shape=(total_samples, height, width))
        for path in prediction_paths
    ]

    result = _calibrate_decision_threshold(
        optimization.ThresholdCalibrationConfig(
            patient_ids=["p1", "p2", "p3", "p4", "p5"],
            local_map={
                "p1": slice(0, 1),
                "p2": slice(1, 2),
                "p3": slice(2, 3),
                "p4": slice(3, 4),
                "p5": slice(4, 5),
            },
            global_indices=np.array([0, 1, 2, 3, 4]),
            truth_memmap=truth_memmap,
            prediction_memmaps=prediction_memmaps,
            semantic_indices=[0],
            semantic_weights=[1.0],
            spatial_indices=[1],
            spatial_weights=[1.0],
            roi_context_scale=1,
            roi_threshold=0.5,
            optimizer_config=replace(
                optimizer_config,
                decision_threshold_min=0.5,
                decision_threshold_max=0.5,
                decision_threshold_step=0.1,
                pos_dice_drop_tolerance=0.02,
                min_component_area_px_candidates=(0, RULE6_COMPONENT_AREA_CANDIDATE),
                min_patient_positive_patches_candidates=(1,),
            ),
        )
    )

    assert result.postprocessing_config.min_component_area_px == 0
    assert result.metrics["Calibration_positive_dice"] == pytest.approx(1.0)
    assert result.summary["rejected_candidate_count"] == 1


def test_calibrate_rule6_rejects_positive_tpr_drop(
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    positive_truth = np.array(
        [
            [
                [1, 1, 0],
                [1, 1, 0],
                [0, 0, 0],
            ]
        ],
        dtype=np.uint8,
    )
    positive_prediction = np.array(
        [
            [
                [0.9, 0.9, 0.6],
                [0.9, 0.6, 0.6],
                [0.6, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    negative_truth = np.zeros((1, 3, 3), dtype=np.uint8)
    negative_prediction = np.array(
        [
            [
                [0.6, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    roi_mask = np.ones_like(positive_truth, dtype=np.uint8)
    patient_predictions = [
        ("positive", positive_prediction, positive_truth, roi_mask),
        ("negative", negative_prediction, negative_truth, roi_mask),
    ]
    config = replace(
        optimizer_config,
        decision_threshold_min=0.5,
        decision_threshold_max=0.8,
        decision_threshold_step=0.3,
        pos_dice_drop_tolerance=1.0,
        pos_tpr_drop_tolerance=0.05,
        min_component_area_px_candidates=(0,),
        min_patient_positive_patches_candidates=(1,),
    )

    strict_result = optimization._calibrate_rule6_from_patient_predictions(
        optimizer_config=config,
        patient_predictions=patient_predictions,
    )
    relaxed_result = optimization._calibrate_rule6_from_patient_predictions(
        optimizer_config=replace(config, pos_tpr_drop_tolerance=0.50),
        patient_predictions=patient_predictions,
    )

    assert strict_result.decision_threshold == pytest.approx(0.5)
    assert strict_result.metrics["Calibration_baseline_positive_tpr"] == pytest.approx(1.0)
    assert strict_result.metrics["Calibration_positive_tpr"] == pytest.approx(1.0)
    assert strict_result.summary["minimum_allowed_positive_tpr"] == pytest.approx(0.95)
    assert strict_result.summary["rejected_candidate_count"] == 1
    assert relaxed_result.decision_threshold == pytest.approx(0.8)
    assert relaxed_result.metrics["Calibration_positive_tpr"] == pytest.approx(0.75)


def test_calibrate_rule6_uses_patient_area_suppression_for_tiny_false_positive(
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    positive_truth = np.zeros((1, 10, 10), dtype=np.uint8)
    positive_truth[:, :2, :2] = 1
    positive_prediction = positive_truth.astype(np.float32) * 0.9
    negative_truth = np.zeros((1, 10, 10), dtype=np.uint8)
    negative_prediction = np.zeros((1, 10, 10), dtype=np.float32)
    negative_prediction[:, 0, 0] = 0.9
    roi_mask = np.ones_like(positive_truth, dtype=np.uint8)

    result = optimization._calibrate_rule6_from_patient_predictions(
        optimizer_config=replace(
            optimizer_config,
            decision_threshold_min=0.5,
            decision_threshold_max=0.5,
            decision_threshold_step=0.1,
            min_micro_dice=V3_NEAR_MICRO_DICE_FLOOR,
            negative_clean_target=0.50,
            min_macro_precision=0.50,
            min_macro_tpr=0.80,
            min_component_area_px_candidates=(0,),
            min_patient_positive_patches_candidates=(1,),
            min_patient_positive_area_fraction_candidates=(0.0, 0.02),
        ),
        patient_predictions=[
            ("positive", positive_prediction, positive_truth, roi_mask),
            ("negative", negative_prediction, negative_truth, roi_mask),
        ],
    )

    assert result.postprocessing_config.min_patient_positive_area_fraction == pytest.approx(0.02)
    assert result.metrics["Calibration_target_status"] == "target_met"
    assert result.metrics["Calibration_negative_clean_rate"] == pytest.approx(1.0)
    assert result.metrics["Calibration_micro_dice"] == pytest.approx(1.0)


def test_calibrate_rule6_reports_target_infeasible_when_guardrails_cannot_be_met(
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    positive_truth = np.zeros((1, 10, 10), dtype=np.uint8)
    positive_truth[:, :2, :2] = 1
    positive_prediction = positive_truth.astype(np.float32) * 0.9
    negative_truth = np.zeros((1, 10, 10), dtype=np.uint8)
    negative_prediction = np.zeros((1, 10, 10), dtype=np.float32)
    negative_prediction[:, 0, 0] = 0.9
    roi_mask = np.ones_like(positive_truth, dtype=np.uint8)

    result = optimization._calibrate_rule6_from_patient_predictions(
        optimizer_config=replace(
            optimizer_config,
            decision_threshold_min=0.5,
            decision_threshold_max=0.5,
            decision_threshold_step=0.1,
            min_micro_dice=V3_NEAR_MICRO_DICE_FLOOR,
            negative_clean_target=1.0,
            min_macro_precision=0.50,
            min_macro_tpr=0.80,
            min_component_area_px_candidates=(0,),
            min_patient_positive_patches_candidates=(1,),
            min_patient_positive_area_fraction_candidates=(0.0,),
        ),
        patient_predictions=[
            ("positive", positive_prediction, positive_truth, roi_mask),
            ("negative", negative_prediction, negative_truth, roi_mask),
        ],
    )

    assert result.metrics["Calibration_target_status"] == "target_infeasible"
    assert result.metrics["Calibration_negative_clean_rate"] == pytest.approx(0.0)
    assert float(result.metrics["Calibration_micro_dice"]) >= V3_NEAR_MICRO_DICE_FLOOR


def test_spatial_objective_precompute_matches_full_image_objective(
    tmp_path: Path,
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
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
            np.full((2, 2, 2), UINT16_MAX, dtype=np.uint16),
            np.array(
                [
                    [[UINT16_MAX, 0], [0, 0]],
                    [[UINT16_MAX, UINT16_MAX], [0, 0]],
                ],
                dtype=np.uint16,
            ),
            np.array(
                [
                    [[0, UINT16_MAX], [0, 0]],
                    [[0, UINT16_MAX], [UINT16_MAX, 0]],
                ],
                dtype=np.uint16,
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
    prepared = optimization.OptimizationPreparation(
        patient_map={"p1": [0], "p2": [1]},
        prediction_memmaps=prediction_memmaps,
        truth_memmap=truth_memmap,
        holdout_idx=np.array([], dtype=np.int64),
        holdout_local_map={},
        holdout_patients=[],
        calibration_idx=np.array([], dtype=np.int64),
        calibration_local_map={},
        calibration_patients=[],
        optimization_idx=np.array([0, 1], dtype=np.int64),
        optimization_local_map={"p1": slice(0, 1), "p2": slice(1, 2)},
        optimization_patients=["p1", "p2"],
        semantic_indices=[0],
        spatial_indices=[1, 2],
        optimization_truth=None,
        optimization_semantic_cache=None,
        optimization_spatial_cache=None,
        optimization_truth_cache=None,
        optimization_gt_density_by_patient=None,
        optimization_positive_patients={"p1"},
        height=height,
        width=width,
    )
    config = replace(
        optimizer_config,
        spatial_architectures=("FPN", "MANET"),
        spatial_patient_policy="all",
    )
    fixed_roi_mask = np.array(
        [
            [[1, 1], [0, 0]],
            [[1, 1], [1, 0]],
        ],
        dtype=np.uint8,
    )
    trial = cast(optuna.Trial, _FixedTrial({"w_spa_0": 0.25, "w_spa_1": 0.75}))

    patient_caches = optimization._build_spatial_objective_cache(
        config,
        prepared,
        fixed_roi_mask,
    )
    precomputed_score = optimization._spatial_objective(
        trial,
        config=config,
        prepared=prepared,
        patient_caches=patient_caches,
    )

    weights = _normalize_weights([0.25, 0.75])
    positive_auprc_total = 0.0
    spill_total = 0.0
    negative_fp_total = 0.0
    evaluated_positive_patients = 0
    evaluated_negative_patients = 0
    for patient_id in prepared.optimization_patients:
        patient_truth = _optimization_patient_truth(prepared, patient_id=patient_id)
        local_slice = prepared.optimization_local_map[patient_id]
        patient_roi = fixed_roi_mask[local_slice]
        patient_prediction = optimization._optimization_patient_spatial_prediction(
            prepared,
            patient_id=patient_id,
            weights=weights,
        )
        if patient_id in prepared.optimization_positive_patients:
            positive_auprc_total += compute_patient_auprc_in_roi(
                patient_truth.ravel(),
                patient_prediction.ravel(),
                patient_roi.ravel(),
            )
            evaluated_positive_patients += 1
        else:
            negative_fp_total += float(np.mean(patient_prediction))
            evaluated_negative_patients += 1
        mass_total = float(np.sum(patient_prediction) + 1e-7)
        mass_outside = float(np.sum(patient_prediction * (1 - patient_roi)))
        spill_total += mass_outside / mass_total

    macro_positive_auprc = positive_auprc_total / evaluated_positive_patients
    macro_spill = spill_total / len(prepared.optimization_patients)
    macro_negative_fp = negative_fp_total / evaluated_negative_patients
    expected_score = (
        macro_positive_auprc
        - (config.spill_penalty_lambda * macro_spill)
        - (config.spill_penalty_lambda * macro_negative_fp)
    )

    assert precomputed_score == pytest.approx(expected_score)


def test_semantic_objective_early_prunes_impossible_positive_recall(
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    prepared = optimization.OptimizationPreparation(
        patient_map={"p1": [0], "p2": [1]},
        prediction_memmaps=[],
        truth_memmap=cast(Any, np.ones((2, 2, 2), dtype=np.uint8)),
        holdout_idx=np.array([], dtype=np.int64),
        holdout_local_map={},
        holdout_patients=[],
        calibration_idx=np.array([], dtype=np.int64),
        calibration_local_map={},
        calibration_patients=[],
        optimization_idx=np.array([0, 1], dtype=np.int64),
        optimization_local_map={"p1": slice(0, 1), "p2": slice(1, 2)},
        optimization_patients=["p1", "p2"],
        semantic_indices=[0],
        spatial_indices=[0],
        optimization_truth=None,
        optimization_semantic_cache={
            "p1": np.asarray([np.zeros((1, 2, 2), dtype=np.uint16)]),
            "p2": np.asarray([np.full((1, 2, 2), UINT16_MAX, dtype=np.uint16)]),
        },
        optimization_spatial_cache=None,
        optimization_truth_cache={
            "p1": np.ones((1, 2, 2), dtype=np.uint8),
            "p2": np.ones((1, 2, 2), dtype=np.uint8),
        },
        optimization_gt_density_by_patient={"p1": 1.0, "p2": 1.0},
        optimization_positive_patients={"p1", "p2"},
        height=2,
        width=2,
    )
    config = replace(optimizer_config, roi_min_pos_recall=0.8, roi_empty_max=1.0)
    trial = cast(optuna.Trial, _FixedTrial({"w_sem_0": 1.0, "roi_thresh": 0.5}))

    with pytest.raises(optuna.exceptions.TrialPruned, match="Misses Positives"):
        optimization._semantic_objective(trial, config=config, prepared=prepared)


def test_semantic_objective_early_checks_preserve_completed_trial_score(
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    prepared = optimization.OptimizationPreparation(
        patient_map={"p1": [0], "p2": [1]},
        prediction_memmaps=[],
        truth_memmap=cast(Any, np.ones((2, 2, 2), dtype=np.uint8)),
        holdout_idx=np.array([], dtype=np.int64),
        holdout_local_map={},
        holdout_patients=[],
        calibration_idx=np.array([], dtype=np.int64),
        calibration_local_map={},
        calibration_patients=[],
        optimization_idx=np.array([0, 1], dtype=np.int64),
        optimization_local_map={"p1": slice(0, 1), "p2": slice(1, 2)},
        optimization_patients=["p1", "p2"],
        semantic_indices=[0],
        spatial_indices=[0],
        optimization_truth=None,
        optimization_semantic_cache={
            "p1": np.asarray([np.full((1, 2, 2), UINT16_MAX, dtype=np.uint16)]),
            "p2": np.asarray([np.full((1, 2, 2), UINT16_MAX, dtype=np.uint16)]),
        },
        optimization_spatial_cache=None,
        optimization_truth_cache={
            "p1": np.ones((1, 2, 2), dtype=np.uint8),
            "p2": np.ones((1, 2, 2), dtype=np.uint8),
        },
        optimization_gt_density_by_patient={"p1": 1.0, "p2": 1.0},
        optimization_positive_patients={"p1", "p2"},
        height=2,
        width=2,
    )
    trial = cast(optuna.Trial, _FixedTrial({"w_sem_0": 1.0, "roi_thresh": 0.5}))

    score = optimization._semantic_objective(trial, config=optimizer_config, prepared=prepared)

    assert score == pytest.approx(1.0)


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
    assert prediction_memmap[0, 1, 0] == UINT16_MAX
    assert prediction_memmap[0, 1, 1] == 0
    assert prediction_memmap[1, 0, 0] == UINT16_MAX
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


def test_study_progress_callback_ignores_all_pruned_study() -> None:
    class _RecordingProgressBar:
        def __init__(self) -> None:
            self.updated = 0
            self.postfix: dict[str, str] | None = None

        def update(self, value: int) -> None:
            self.updated += value

        def set_postfix(self, postfix: dict[str, str]) -> None:
            self.postfix = postfix

    progress_bar = _RecordingProgressBar()
    callback = optimization._StudyProgressCallback(cast(Any, progress_bar))
    pruned_trial = cast(
        optuna.trial.FrozenTrial,
        SimpleNamespace(state=optuna.trial.TrialState.PRUNED),
    )
    study = cast(Any, SimpleNamespace(trials=[pruned_trial]))

    callback(study, pruned_trial)

    assert progress_bar.updated == 1
    assert progress_bar.postfix == {"done": "0", "pruned": "1"}


def test_run_two_stream_optimization_raises_clear_error_when_semantic_trials_all_pruned(
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
            np.array([0])
            if selected_patients == {"p1"}
            else (np.array([], dtype=np.int64) if not selected_patients else np.array([1])),
            {"p1": slice(0, 1)}
            if selected_patients == {"p1"}
            else ({} if not selected_patients else {"p2": slice(0, 1)}),
            ["p1"] if selected_patients == {"p1"} else ([] if not selected_patients else ["p2"]),
        ),
    )

    class _AllPrunedStudy:
        def __init__(self) -> None:
            self.trials = [SimpleNamespace(state=optuna.trial.TrialState.PRUNED)]

        def optimize(
            self, objective: Any, n_trials: int, callbacks: list[Any] | None = None
        ) -> None:
            del objective, n_trials, callbacks

    optuna_module = optimization.optuna  # type: ignore[attr-defined]
    monkeypatch.setattr(
        optuna_module,
        "create_study",
        lambda direction, sampler: _AllPrunedStudy(),
    )

    with pytest.raises(
        RuntimeError,
        match="Semantic optimization produced no completed trials; all trials were pruned",
    ):
        run_two_stream_optimization(
            optimizer_config,
            [_ConstantBinaryModel(0.0, arch_name="SWIN")],
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
            np.array([0])
            if selected_patients == {"p1"}
            else (np.array([], dtype=np.int64) if not selected_patients else np.array([1])),
            {"p1": slice(0, 1)}
            if selected_patients == {"p1"}
            else ({} if not selected_patients else {"p2": slice(0, 1)}),
            ["p1"] if selected_patients == {"p1"} else ([] if not selected_patients else ["p2"]),
        ),
    )

    class _FakeStudy:
        def __init__(self, best_params: dict[str, float]) -> None:
            self.best_params = best_params
            self.best_value = 0.0
            self.trials: list[Any] = [SimpleNamespace(state=optuna.trial.TrialState.COMPLETE)]

        def optimize(
            self, objective: Any, n_trials: int, callbacks: list[Any] | None = None
        ) -> None:
            del objective, n_trials, callbacks

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


def test_run_two_stream_optimization_staged_cleans_raw_prediction_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optimizer_config: EnsembleOptimizerConfig,
) -> None:
    patient_truths = {
        "p1": np.array([[1, 0], [0, 0]], dtype=np.uint8),
        "p2": np.zeros((2, 2), dtype=np.uint8),
        "p3": np.array([[1, 0], [0, 0]], dtype=np.uint8),
        "p4": np.array([[1, 0], [0, 0]], dtype=np.uint8),
    }
    factory_calls: list[set[str] | None] = []

    def dataloader_factory(allowed_patients: set[str] | None) -> _ListLoader:
        factory_calls.append(set(allowed_patients) if allowed_patients is not None else None)
        selected = sorted(allowed_patients or set(patient_truths))
        if not selected:
            return _ListLoader(dataset_len=0, batches=[])
        images = torch.zeros((len(selected), 3, 2, 2), dtype=torch.uint8)
        masks = torch.zeros((len(selected), 2, 2, 2), dtype=torch.uint8)
        for index, patient_id in enumerate(selected):
            masks[index, 1] = torch.from_numpy(patient_truths[patient_id])
        return _ListLoader(dataset_len=len(selected), batches=[(images, masks, selected)])

    monkeypatch.setattr(
        optimization,
        "predict_with_tta_batched",
        lambda model, images, architecture: torch.full(
            (images.shape[0], images.shape[2], images.shape[3]),
            0.8 if architecture == "SWIN" else 0.6,
            dtype=torch.float32,
        ),
    )
    monkeypatch.setattr(optimization, "clear_gpu", lambda: None)

    class _FakeStudy:
        def __init__(self, best_params: dict[str, float]) -> None:
            self.best_params = best_params
            self.best_value = 1.0
            self.trials: list[Any] = [SimpleNamespace(state=optuna.trial.TrialState.COMPLETE)]

        def optimize(
            self, objective: Any, n_trials: int, callbacks: list[Any] | None = None
        ) -> None:
            del objective, n_trials
            if callbacks:
                for callback in callbacks:
                    callback(self, cast(Any, object()))

    studies = iter([_FakeStudy({"w_sem_0": 1.0, "roi_thresh": 0.5}), _FakeStudy({"w_spa_0": 1.0})])
    optuna_module = optimization.optuna  # type: ignore[attr-defined]
    monkeypatch.setattr(optuna_module, "create_study", lambda direction, sampler: next(studies))

    result = run_two_stream_optimization(
        optimizer_config,
        [_ConstantBinaryModel(0.0, arch_name="SWIN"), _TupleTwoClassModel(arch_name="FPN")],
        cast(Any, _ListLoader(0, [])),
        device=torch.device("cpu"),
        predefined_split=HoldoutSplit(
            optimization_patients={"p1"},
            calibration_patients={"p2", "p3"},
            holdout_patients={"p4"},
            positive_patients={"p1", "p3", "p4"},
            negative_patients={"p2"},
        ),
        dataloader_factory=cast(Any, dataloader_factory),
    )

    assert result.semantic_indices == [0]
    assert result.spatial_indices == [1]
    assert {path.name for path in optimizer_config.pred_cache_dir.rglob("*.dat")} == set()
    assert {"p1"} in factory_calls
    assert {"p2"} in factory_calls
    assert {"p3"} in factory_calls
    assert {"p4"} in factory_calls
    assert None not in factory_calls


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
            self.best_value = 0.0
            self.trials: list[Any] = [SimpleNamespace(state=optuna.trial.TrialState.COMPLETE)]

        def optimize(
            self, objective: Any, n_trials: int, callbacks: list[Any] | None = None
        ) -> None:
            del objective, n_trials, callbacks

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
    assert result.roi_threshold == DEFAULT_THRESHOLD
    assert result.decision_threshold == DEFAULT_THRESHOLD
    assert result.postprocessing_config.min_component_area_px == 0
    assert result.postprocessing_config.min_patient_positive_patches == 1
    assert result.calibration_metrics["Calibration_metric"] == "Macro_Rule6_Dice"
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
            self.best_value = 0.0
            self.trials: list[Any] = [SimpleNamespace(state=optuna.trial.TrialState.COMPLETE)]

        def optimize(
            self, objective: Any, n_trials: int, callbacks: list[Any] | None = None
        ) -> None:
            del n_trials
            self.best_value = float(objective(_FakeTrial()))
            if callbacks:
                for callback in callbacks:
                    callback(self, cast(Any, object()))

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
    assert result.holdout_metrics["N_eval_patients"] == EVAL_PATIENT_COUNT
    assert result.holdout_metrics["N_eval_negative_patients"] == 1
    assert result.holdout_metrics["Macro_AUPRC_in_ROI_Positive"] == pytest.approx(0.8)
    assert result.holdout_metrics["Macro_Negative_FP"] == pytest.approx(1.0)
    assert result.holdout_metrics["Macro_Spill_All"] == pytest.approx(0.0)
    assert result.holdout_metrics["Objective_Composite"] == pytest.approx(0.7)
