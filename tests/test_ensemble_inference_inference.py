from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from helpers.ensemble_inference import inference
from helpers.ensemble_postprocessing import PostprocessingConfig
from helpers.runtime_platform import COLAB_INLINE_MATPLOTLIB_BACKEND, HEADLESS_MATPLOTLIB_BACKEND

ANALYSIS_SEED = 17
EXPECTED_AUC = 0.75
SECOND_SAMPLE_ID = 2
THIRD_SAMPLE_ID = 3
VISUALIZATION_SAMPLE_COUNT = 2


def _analysis_config() -> inference.EnsembleAnalysisConfig:
    return inference.EnsembleAnalysisConfig(
        device=torch.device("cpu"),
        roi_threshold=0.33,
        decision_threshold=0.67,
        postprocessing_config=PostprocessingConfig(
            min_component_area_px=0,
            min_patient_positive_patches=1,
        ),
        roi_scale=2,
        train_mean=[0.1, 0.2, 0.3],
        train_std=[0.4, 0.5, 0.6],
        gpu_normalizer=_IdentityNormalizer(),
        seed=ANALYSIS_SEED,
    )


def _visualization_config(
    tmp_path: Path, *, num_samples: int
) -> inference.VisualizationExportConfig:
    return inference.VisualizationExportConfig(
        device=torch.device("cpu"),
        roi_threshold=0.5,
        decision_threshold=0.5,
        postprocessing_config=PostprocessingConfig(
            min_component_area_px=0,
            min_patient_positive_patches=1,
        ),
        roi_scale=2,
        train_mean=[0.1, 0.2, 0.3],
        train_std=[0.4, 0.5, 0.6],
        constituent_models_info=[],
        gpu_normalizer=_IdentityNormalizer(),
        output_dir=tmp_path,
        num_samples=num_samples,
    )


class _ConstantBinaryModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (images.shape[0], 1, images.shape[2], images.shape[3]),
            self.value,
            dtype=torch.float32,
            device=images.device,
        )


class _TupleTwoClassModel(nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.zeros(
            (images.shape[0], 2, images.shape[2], images.shape[3]),
            dtype=torch.float32,
            device=images.device,
        )
        output[:, 1] = 1.0
        aux = torch.zeros((), dtype=torch.float32, device=images.device)
        return output, aux


class _IdentityNormalizer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()


def test_autocast_context_returns_nullcontext_on_cpu() -> None:
    context = inference._autocast_context(torch.zeros((1, 3, 2, 2)), use_amp=True)

    assert isinstance(context, contextlib.nullcontext)


def test_predict_with_tta_batched_handles_single_channel_logits() -> None:
    images = torch.zeros((2, 3, 4, 4), dtype=torch.float32)

    probabilities = inference.predict_with_tta_batched(_ConstantBinaryModel(0.0), images)

    assert torch.allclose(probabilities, torch.full((2, 4, 4), 0.5))


def test_predict_with_tta_batched_handles_two_class_tuple_output() -> None:
    images = torch.zeros((1, 3, 3, 3), dtype=torch.float32)

    probabilities = inference.predict_with_tta_batched(_TupleTwoClassModel(), images)
    expected_prob = float(torch.softmax(torch.tensor([0.0, 1.0]), dim=0)[1])

    assert torch.allclose(
        probabilities,
        torch.full((1, 3, 3), expected_prob),
    )


def test_compute_two_stream_probabilities_returns_zero_mask_when_stream_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference,
        "predict_with_tta_batched",
        lambda model, images, use_amp=True: torch.ones(
            (images.shape[0], images.shape[2], images.shape[3])
        ),
    )

    result = inference.compute_two_stream_probabilities(
        [nn.Identity()],
        [{"stream_role": "semantic", "weight": 1.0}],
        torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        roi_threshold=0.5,
        roi_scale=2,
    )

    assert torch.equal(result, torch.zeros((1, 4, 4)))


def test_compute_two_stream_probabilities_applies_roi_gating_and_weight_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel(nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    def fake_predict(model: FakeModel, images: torch.Tensor, use_amp: bool = True) -> torch.Tensor:
        del images, use_amp
        if model.name == "semantic":
            return torch.tensor([[[1.0, 1.0], [0.0, 0.0]]], dtype=torch.float32)
        if model.name == "spatial":
            return torch.tensor([[[0.8, 0.6], [0.4, 0.2]]], dtype=torch.float32)
        raise AssertionError("unexpected model")

    monkeypatch.setattr(inference, "predict_with_tta_batched", fake_predict)

    result = inference.compute_two_stream_probabilities(
        [FakeModel("semantic"), FakeModel("ignored"), FakeModel("spatial")],
        [
            {"stream_role": "semantic", "weight": 1.0},
            {"stream_role": "spatial", "weight": 0.0},
            {"stream_role": "spatial", "weight": 1.0},
        ],
        torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        roi_threshold=0.5,
        roi_scale=1,
    )

    assert torch.equal(result, torch.tensor([[[0.8, 0.6], [0.0, 0.0]]], dtype=torch.float32))


def test_compute_two_stream_probabilities_sanitizes_nan_and_inf_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel(nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    def fake_predict(model: FakeModel, images: torch.Tensor, use_amp: bool = True) -> torch.Tensor:
        del images, use_amp
        if model.name == "semantic":
            return torch.ones((1, 2, 2), dtype=torch.float32)
        return torch.tensor([[[float("nan"), float("inf")], [float("-inf"), 0.25]]])

    monkeypatch.setattr(inference, "predict_with_tta_batched", fake_predict)

    result = inference.compute_two_stream_probabilities(
        [FakeModel("semantic"), FakeModel("spatial")],
        [
            {"stream_role": "semantic", "weight": 1.0},
            {"stream_role": "spatial", "weight": 1.0},
        ],
        torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        roi_threshold=0.1,
        roi_scale=1,
    )

    assert torch.equal(result, torch.tensor([[[0.0, 1.0], [0.0, 0.25]]], dtype=torch.float32))


def test_analyze_ensemble_metrics_skips_none_batches_and_builds_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_summary(
        stats_by_patient: dict[str, list[dict[str, int]]], *, seed: int
    ) -> dict[str, Any]:
        captured["stats"] = stats_by_patient
        captured["seed"] = seed
        return {"base": True}

    monkeypatch.setattr(
        inference,
        "compute_two_stream_probabilities",
        lambda models, meta, images, roi_threshold, roi_scale: torch.tensor(
            [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        ),
    )
    monkeypatch.setattr(inference, "summarize_patient_metrics", fake_summary)
    monkeypatch.setattr(inference, "compute_auc_from_histograms", lambda pos, neg: 0.75)

    test_loader = cast(
        Any,
        [
            None,
            (
                torch.zeros((2, 3, 2, 2), dtype=torch.uint8),
                torch.tensor(
                    [
                        [[0, 1], [1, 0]],
                        [[1, 0], [0, 0]],
                    ],
                    dtype=torch.uint8,
                ),
                ["patient-a", "patient-b"],
                ["a.png", "b.png"],
            ),
        ],
    )

    summary = inference.analyze_ensemble_metrics(
        [nn.Identity()],
        [{"stream_role": "semantic", "weight": 1.0}],
        test_loader,
        _analysis_config(),
    )

    assert captured["seed"] == ANALYSIS_SEED
    assert captured["stats"] == {
        "patient-a": [{"tn": 2, "fn": 0, "fp": 0, "tp": 2}],
        "patient-b": [{"tn": 2, "fn": 0, "fp": 1, "tp": 1}],
    }
    assert summary["auc"] == EXPECTED_AUC
    assert summary["auc_source"] == "raw_probabilities_before_hard_postprocessing"
    assert summary["postprocessing"]["min_component_area_px"] == 0
    assert summary["postprocessing"]["min_patient_positive_patches"] == 1
    assert summary["normalization"] == {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}
    assert summary["ensemble"] == {
        "method": "two_stream_spatial_gating",
        "roi_threshold": 0.33,
        "decision_threshold": 0.67,
        "postprocessing": {
            "method": "threshold_components_patient_suppression",
            "min_component_area_px": 0,
            "min_patient_positive_patches": 1,
        },
        "weights": None,
    }


def test_analyze_ensemble_metrics_uses_declared_threshold_for_hard_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_summary(
        stats_by_patient: dict[str, list[dict[str, int]]], *, seed: int
    ) -> dict[str, Any]:
        captured["stats"] = stats_by_patient
        captured["seed"] = seed
        return {}

    monkeypatch.setattr(
        inference,
        "compute_two_stream_probabilities",
        lambda models, meta, images, roi_threshold, roi_scale: torch.tensor(
            [[[0.2, 0.8], [0.4, 0.6]]],
            dtype=torch.float32,
        ),
    )
    monkeypatch.setattr(inference, "summarize_patient_metrics", fake_summary)
    monkeypatch.setattr(inference, "compute_auc_from_histograms", lambda pos, neg: 0.5)

    test_loader = cast(
        Any,
        [
            (
                torch.zeros((1, 3, 2, 2), dtype=torch.uint8),
                torch.tensor([[[0, 1], [1, 0]]], dtype=torch.uint8),
                ["patient-threshold"],
                ["threshold.png"],
            ),
        ],
    )

    inference.analyze_ensemble_metrics(
        [nn.Identity()],
        [{"stream_role": "semantic", "weight": 1.0}],
        test_loader,
        inference.EnsembleAnalysisConfig(
            device=torch.device("cpu"),
            roi_threshold=0.25,
            decision_threshold=0.5,
            postprocessing_config=PostprocessingConfig(
                min_component_area_px=0,
                min_patient_positive_patches=1,
            ),
            roi_scale=2,
            train_mean=[0.1, 0.2, 0.3],
            train_std=[0.4, 0.5, 0.6],
            gpu_normalizer=_IdentityNormalizer(),
            seed=ANALYSIS_SEED,
        ),
    )

    assert captured["seed"] == ANALYSIS_SEED
    assert captured["stats"] == {"patient-threshold": [{"tn": 1, "fn": 1, "fp": 1, "tp": 1}]}


def test_analyze_ensemble_metrics_applies_frozen_patient_suppression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_summary(
        stats_by_patient: dict[str, list[dict[str, int]]], *, seed: int
    ) -> dict[str, Any]:
        del seed
        captured["stats"] = stats_by_patient
        return {}

    monkeypatch.setattr(
        inference,
        "compute_two_stream_probabilities",
        lambda models, meta, images, roi_threshold, roi_scale: torch.tensor(
            [
                [[0.9, 0.1], [0.1, 0.1]],
                [[0.1, 0.1], [0.1, 0.1]],
            ],
            dtype=torch.float32,
        ),
    )
    monkeypatch.setattr(inference, "summarize_patient_metrics", fake_summary)
    monkeypatch.setattr(inference, "compute_auc_from_histograms", lambda pos, neg: 0.5)

    test_loader = cast(
        Any,
        [
            (
                torch.zeros((2, 3, 2, 2), dtype=torch.uint8),
                torch.tensor(
                    [
                        [[1, 0], [0, 0]],
                        [[0, 0], [0, 0]],
                    ],
                    dtype=torch.uint8,
                ),
                ["patient-suppress", "patient-suppress"],
                ["a.png", "b.png"],
            ),
        ],
    )

    inference.analyze_ensemble_metrics(
        [nn.Identity()],
        [{"stream_role": "semantic", "weight": 1.0}],
        test_loader,
        inference.EnsembleAnalysisConfig(
            device=torch.device("cpu"),
            roi_threshold=0.25,
            decision_threshold=0.5,
            postprocessing_config=PostprocessingConfig(
                min_component_area_px=0,
                min_patient_positive_patches=2,
            ),
            roi_scale=2,
            train_mean=[0.1, 0.2, 0.3],
            train_std=[0.4, 0.5, 0.6],
            gpu_normalizer=_IdentityNormalizer(),
            seed=ANALYSIS_SEED,
        ),
    )

    assert captured["stats"] == {"patient-suppress": [{"tp": 0, "fp": 0, "fn": 1, "tn": 7}]}


def test_export_visualizations_returns_empty_for_nonpositive_sample_count(tmp_path: Path) -> None:
    output_paths = inference.export_visualizations(
        [nn.Identity()],
        cast(Any, []),
        _visualization_config(tmp_path, num_samples=0),
    )

    assert output_paths == []


def test_export_visualizations_returns_empty_for_empty_or_none_batch(tmp_path: Path) -> None:
    assert (
        inference.export_visualizations(
            [nn.Identity()],
            cast(Any, []),
            _visualization_config(tmp_path / "empty", num_samples=1),
        )
        == []
    )
    assert (
        inference.export_visualizations(
            [nn.Identity()],
            cast(Any, [None]),
            _visualization_config(tmp_path / "none", num_samples=1),
        )
        == []
    )


def test_export_visualizations_writes_requested_number_of_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLBACKEND", COLAB_INLINE_MATPLOTLIB_BACKEND)
    monkeypatch.setattr(
        inference,
        "compute_two_stream_probabilities",
        lambda models, meta, images, roi_threshold, roi_scale: torch.tensor(
            [[[0.9, 0.1], [0.8, 0.2]], [[0.2, 0.3], [0.4, 0.9]]],
            dtype=torch.float32,
        ),
    )
    dataloader = cast(
        Any,
        [
            (
                torch.ones((2, 3, 2, 2), dtype=torch.float32),
                torch.tensor(
                    [
                        [[0, 1], [1, 0]],
                        [[1, 0], [0, 1]],
                    ],
                    dtype=torch.uint8,
                ),
                ["p1", "p2"],
                ["f1.png", "f2.png"],
            )
        ],
    )

    output_paths = inference.export_visualizations(
        [nn.Identity()],
        dataloader,
        inference.VisualizationExportConfig(
            device=torch.device("cpu"),
            roi_threshold=0.5,
            decision_threshold=0.5,
            postprocessing_config=PostprocessingConfig(
                min_component_area_px=0,
                min_patient_positive_patches=1,
            ),
            roi_scale=2,
            train_mean=[0.1, 0.2, 0.3],
            train_std=[0.4, 0.5, 0.6],
            constituent_models_info=[{"stream_role": "semantic", "weight": 1.0}],
            gpu_normalizer=_IdentityNormalizer(),
            output_dir=tmp_path,
            num_samples=1,
        ),
    )

    assert len(output_paths) == 1
    assert output_paths[0].exists()
    assert output_paths[0].name.startswith("worst_dice_01__")
    assert os.environ["MPLBACKEND"] == HEADLESS_MATPLOTLIB_BACKEND


def test_export_visualizations_ranks_worst_dice_across_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_compute(
        models: Any, meta: Any, images: torch.Tensor, roi_threshold: float, roi_scale: int
    ) -> torch.Tensor:
        del models, meta, roi_threshold, roi_scale
        outputs: list[torch.Tensor] = []
        for image in images:
            sample_id = int(image[0, 0, 0].item())
            if sample_id == 1:
                outputs.append(torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32))
            elif sample_id == SECOND_SAMPLE_ID:
                outputs.append(torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32))
            elif sample_id == THIRD_SAMPLE_ID:
                outputs.append(torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32))
            else:
                raise AssertionError(f"unexpected sample id: {sample_id}")
        return torch.stack(outputs)

    monkeypatch.setattr(inference, "compute_two_stream_probabilities", fake_compute)
    dataloader = cast(
        Any,
        [
            (
                torch.tensor(
                    [
                        [[[1, 1], [1, 1]]] * 3,
                        [[[2, 2], [2, 2]]] * 3,
                    ],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        [[1, 0], [0, 0]],
                        [[1, 0], [0, 0]],
                    ],
                    dtype=torch.uint8,
                ),
                ["p1", "p2"],
                ["f1.png", "f2.png"],
            ),
            (
                torch.tensor([[[[3, 3], [3, 3]]] * 3], dtype=torch.float32),
                torch.tensor([[[1, 1], [0, 0]]], dtype=torch.uint8),
                ["p3"],
                ["f3.png"],
            ),
        ],
    )

    output_paths = inference.export_visualizations(
        [nn.Identity()],
        dataloader,
        inference.VisualizationExportConfig(
            device=torch.device("cpu"),
            roi_threshold=0.5,
            decision_threshold=0.5,
            postprocessing_config=PostprocessingConfig(
                min_component_area_px=0,
                min_patient_positive_patches=1,
            ),
            roi_scale=2,
            train_mean=[0.1, 0.2, 0.3],
            train_std=[0.4, 0.5, 0.6],
            constituent_models_info=[{"stream_role": "semantic", "weight": 1.0}],
            gpu_normalizer=_IdentityNormalizer(),
            output_dir=tmp_path,
            num_samples=VISUALIZATION_SAMPLE_COUNT,
        ),
    )

    assert [path.name for path in output_paths] == [
        "worst_dice_01__p1__f1.png",
        "worst_dice_02__p2__f2.png",
    ]
    assert all(path.exists() for path in output_paths)
