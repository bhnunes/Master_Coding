# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from helpers.lr_finder import runner as runner_module
from helpers.lr_finder.config import BCEDiceSearchSpace, LRFinderConfig, ModelPlan
from helpers.lr_finder.reporting import RunRecord
from helpers.lr_finder.runner import (
    LossConfigRunContext,
    LRFinderRunConfig,
    _run_single_loss_config,
    clear_gpu,
    configure_execution_mode,
    plot_stability_curves,
    run_lr_finder_once,
    run_lr_finder_screening,
)
from helpers.lr_finder.search_space import BCEDiceParams
from helpers.training.gpu import GPUDownscale, GPUNormalizer


def _build_config(tmp_path: Path) -> LRFinderConfig:
    return LRFinderConfig(
        master_manifest_path=tmp_path / "master_manifest.sqlite",
        output_dir=tmp_path / "reports",
        local_data_dir=tmp_path / "local",
        stage_input_locally=False,
        overwrite_output=True,
        smart_sampling=True,
        execution_mode="PAPER",
        amp_precision="fp16",
        seed=24,
        batch_size=2,
        workers=0,
        use_subset=False,
        subset_ratio=1.0,
        num_lhs_samples=2,
        end_lr=0.1,
        num_iter=5,
        num_repeats=2,
        optimizer_weight_decay=1e-4,
        optimizer_start_lr=1e-8,
        pdf_name="report.pdf",
        hf_token=None,
        search_space=BCEDiceSearchSpace(),
        model_plans=[ModelPlan(architecture="FPN", encoder="resnet34")],
    )


def _run_config(**overrides: Any) -> LRFinderRunConfig:
    payload = {
        "model": cast(Any, SimpleNamespace()),
        "optimizer": cast(Any, SimpleNamespace()),
        "criterion": cast(Any, SimpleNamespace()),
        "train_loader": [],
        "device": cast(Any, "cpu"),
        "end_lr": 0.1,
        "num_iter": 5,
        "architecture": "FPN",
        "amp_precision": "fp16",
        "gpu_normalizer": cast(GPUNormalizer, lambda x: x),
        "gpu_downscale": cast(GPUDownscale, lambda x: x),
    }
    payload.update(overrides)
    return LRFinderRunConfig(**payload)


def _loss_context(
    *,
    data_bundle: Any,
    model_plan: ModelPlan,
    params: BCEDiceParams,
    config_index: int,
    device: object = "cpu",
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> LossConfigRunContext:
    return LossConfigRunContext(
        device=cast(Any, device),
        data_bundle=data_bundle,
        model_plan=model_plan,
        params=params,
        config_index=config_index,
        gpu_normalizer=cast(GPUNormalizer, lambda x: x),
        gpu_downscale=cast(GPUDownscale, lambda x: x),
        initial_state_dict=initial_state_dict or {},
    )


def test_configure_execution_mode_switches_determinism_flags() -> None:
    configure_execution_mode("FAST_DEV")
    assert cast(Any, __import__("torch")).backends.cudnn.benchmark is True
    assert cast(Any, __import__("torch")).backends.cudnn.deterministic is False

    configure_execution_mode("PAPER")
    assert cast(Any, __import__("torch")).backends.cudnn.benchmark is False
    assert cast(Any, __import__("torch")).backends.cudnn.deterministic is True


def test_clear_gpu_calls_optional_ipc_collect(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
    )
    monkeypatch.setattr("helpers.lr_finder.runner.torch.cuda", fake_cuda)

    clear_gpu()

    assert calls == ["empty_cache", "ipc_collect"]


def test_run_lr_finder_once_raises_when_range_test_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLRFinder:
        def __init__(
            self, model: object, optimizer: object, criterion: object, device: object
        ) -> None:
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device
            self.history: dict[str, list[float]] | None = None

        def range_test(
            self, train_loader: object, end_lr: float, num_iter: int, step_mode: str
        ) -> None:
            del train_loader, end_lr, num_iter, step_mode
            raise RuntimeError("range test failed")

    monkeypatch.setattr(
        "helpers.lr_finder.runner.setup_precision",
        lambda architecture, amp_precision: (None, None, None),
    )
    monkeypatch.setattr(
        "helpers.lr_finder.runner.autocast_ctx",
        lambda images, amp_dtype: nullcontext(),
    )
    import sys

    monkeypatch.setitem(sys.modules, "torch_lr_finder", SimpleNamespace(LRFinder=FakeLRFinder))

    with pytest.raises(RuntimeError, match="range test failed"):
        run_lr_finder_once(_run_config())


def test_run_lr_finder_once_returns_empty_arrays_when_history_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLRFinder:
        def __init__(
            self, model: object, optimizer: object, criterion: object, device: object
        ) -> None:
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device
            self.history: dict[str, list[float]] | None = None

        def range_test(
            self, train_loader: object, end_lr: float, num_iter: int, step_mode: str
        ) -> None:
            del train_loader, end_lr, num_iter, step_mode

    monkeypatch.setattr(
        "helpers.lr_finder.runner.setup_precision",
        lambda architecture, amp_precision: (None, None, None),
    )
    monkeypatch.setattr(
        "helpers.lr_finder.runner.autocast_ctx",
        lambda images, amp_dtype: nullcontext(),
    )
    import sys

    monkeypatch.setitem(sys.modules, "torch_lr_finder", SimpleNamespace(LRFinder=FakeLRFinder))

    history = run_lr_finder_once(_run_config())

    assert history["lr"].size == 0
    assert history["loss"].size == 0


def test_run_lr_finder_once_returns_partial_history_after_non_finite_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLRFinder:
        def __init__(
            self, model: object, optimizer: object, criterion: object, device: object
        ) -> None:
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device
            self.history: dict[str, list[float]] = {"lr": [1e-5, 1e-4], "loss": [0.9, 0.7]}

        def range_test(
            self, train_loader: object, end_lr: float, num_iter: int, step_mode: str
        ) -> None:
            del train_loader, end_lr, num_iter, step_mode
            raise runner_module._NonFiniteLossError("LR finder produced a non-finite loss.")

    monkeypatch.setattr(
        "helpers.lr_finder.runner.setup_precision",
        lambda architecture, amp_precision: (None, None, None),
    )
    monkeypatch.setattr(
        "helpers.lr_finder.runner.autocast_ctx",
        lambda images, amp_dtype: nullcontext(),
    )
    import sys

    monkeypatch.setitem(sys.modules, "torch_lr_finder", SimpleNamespace(LRFinder=FakeLRFinder))

    history = run_lr_finder_once(_run_config())

    assert history["lr"].tolist() == [1e-5, 1e-4]
    assert history["loss"].tolist() == [0.9, 0.7]


def test_run_lr_finder_once_supports_non_blocking_transfer_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def train(self) -> None:
            return None

        def __call__(self, images: torch.Tensor) -> torch.Tensor:
            batch_size, _, height, width = images.shape
            return torch.zeros((batch_size, 2, height, width), dtype=torch.float32)

    class FakeOptimizer:
        def zero_grad(self, set_to_none: bool = True) -> None:
            del set_to_none

        def step(self) -> None:
            return None

    class FakeLoss:
        def __call__(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
            del outputs, masks
            return torch.tensor(0.25, dtype=torch.float32, requires_grad=True)

    class FakeLRFinder:
        def __init__(
            self, model: object, optimizer: object, criterion: object, device: object
        ) -> None:
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device
            self.history: dict[str, list[float]] = {"lr": [], "loss": []}
            self._train_batch: Any

        def range_test(
            self, train_loader: object, end_lr: float, num_iter: int, step_mode: str
        ) -> None:
            del end_lr, step_mode
            train_iter = iter(cast(Any, train_loader))
            for _ in range(num_iter):
                loss = self._train_batch(
                    train_iter,
                    accumulation_steps=1,
                    non_blocking_transfer=True,
                )
                self.history["lr"].append(1e-4)
                self.history["loss"].append(loss)

    monkeypatch.setattr(
        "helpers.lr_finder.runner.setup_precision",
        lambda architecture, amp_precision: (None, None, None),
    )
    monkeypatch.setattr(
        "helpers.lr_finder.runner.autocast_ctx",
        lambda images, amp_dtype: nullcontext(),
    )
    import sys

    monkeypatch.setitem(sys.modules, "torch_lr_finder", SimpleNamespace(LRFinder=FakeLRFinder))

    images = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    masks = torch.zeros((1, 8, 8), dtype=torch.long)
    history = run_lr_finder_once(
        _run_config(
            model=cast(Any, FakeModel()),
            optimizer=cast(Any, FakeOptimizer()),
            criterion=cast(Any, FakeLoss()),
            train_loader=[(images, masks)],
            device=torch.device("cpu"),
            num_iter=1,
        )
    )

    assert history["lr"].tolist() == [1e-4]
    assert history["loss"].tolist() == [0.25]


def test_plot_stability_curves_writes_png(tmp_path: Path) -> None:
    out_png = tmp_path / "plot.png"

    plot_stability_curves(
        [np.array([1e-5, 1e-4, 1e-3, 1e-2], dtype=np.float64)],
        [np.array([5.0, 4.0, 3.0, 2.0], dtype=np.float64)],
        title="Curve",
        out_png=out_png,
        skip_start=1,
        skip_end=1,
    )

    assert out_png.is_file()


def test_run_single_loss_config_returns_none_when_all_repeats_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _build_config(tmp_path)
    model_plan = ModelPlan(architecture="FPN", encoder="resnet34")
    params = BCEDiceParams(alpha=0.1, beta=0.2, gamma=0.3)
    dataset = SimpleNamespace(close=lambda: None)
    data_bundle = SimpleNamespace(dataset=dataset, sample_weights=np.array([1.0], dtype=np.float32))

    monkeypatch.setattr("helpers.lr_finder.runner.seed_everything", lambda seed: None)
    monkeypatch.setattr(
        "helpers.lr_finder.runner.build_train_loader",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "helpers.lr_finder.runner.create_model",
        lambda architecture, encoder, validation=False: (_ for _ in ()).throw(
            RuntimeError("model failed")
        ),
    )
    monkeypatch.setattr("helpers.lr_finder.runner.clear_gpu", lambda: None)

    record, completed, failed = _run_single_loss_config(
        config,
        _loss_context(
            data_bundle=data_bundle,
            model_plan=model_plan,
            params=params,
            config_index=1,
        ),
    )

    assert record is None
    assert completed == 0
    assert failed == config.num_repeats


def test_run_single_loss_config_treats_invalid_curve_stats_as_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _build_config(tmp_path)
    model_plan = ModelPlan(architecture="FPN", encoder="resnet34")
    params = BCEDiceParams(alpha=0.1, beta=0.2, gamma=0.3)
    dataset = SimpleNamespace(close=lambda: None)
    data_bundle = SimpleNamespace(dataset=dataset, sample_weights=np.array([1.0], dtype=np.float32))

    monkeypatch.setattr("helpers.lr_finder.runner.seed_everything", lambda seed: None)
    monkeypatch.setattr(
        "helpers.lr_finder.runner.build_train_loader",
        lambda *args, **kwargs: object(),
    )

    class FakeModel:
        def __init__(self) -> None:
            self.loaded_state_dict: dict[str, torch.Tensor] | None = None

        def to(self, device: object) -> FakeModel:
            del device
            return self

        def parameters(self) -> list[object]:
            return []

        def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
            self.loaded_state_dict = state_dict

    monkeypatch.setattr(
        "helpers.lr_finder.runner.create_model",
        lambda architecture, encoder, validation=False: FakeModel(),
    )
    monkeypatch.setattr(
        "helpers.lr_finder.runner.torch.optim.AdamW",
        lambda params, lr, weight_decay: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "helpers.lr_finder.runner.BCEDiceHybridLossPaper",
        lambda config: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "helpers.lr_finder.runner.run_lr_finder_once",
        lambda config: {
            "lr": np.array([1e-5, 1e-4], dtype=np.float64),
            "loss": np.array([np.nan, np.nan], dtype=np.float64),
        },
    )
    monkeypatch.setattr("helpers.lr_finder.runner.clear_gpu", lambda: None)

    record, completed, failed = _run_single_loss_config(
        config,
        _loss_context(
            data_bundle=data_bundle,
            model_plan=model_plan,
            params=params,
            config_index=1,
            device=torch.device("cpu"),
            initial_state_dict={"weight": torch.tensor([1.0])},
        ),
    )

    assert record is None
    assert completed == 0
    assert failed == config.num_repeats


def test_run_lr_finder_screening_writes_summaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _build_config(tmp_path)
    (config.output_dir / "FPN").mkdir(parents=True, exist_ok=True)
    dataset = SimpleNamespace(closed=False)

    def close_dataset() -> None:
        dataset.closed = True

    dataset.close = close_dataset
    data_bundle = SimpleNamespace(dataset=dataset, sample_weights=np.array([1.0], dtype=np.float32))
    samples = [
        BCEDiceParams(alpha=0.1, beta=0.2, gamma=0.3),
        BCEDiceParams(alpha=0.4, beta=0.5, gamma=0.6),
    ]
    call_count = {"value": 0}
    preload_calls: list[tuple[str, str]] = []

    class DummyProgress:
        def __init__(self) -> None:
            self.updated = 0
            self.finished = False
            self.calls: list[dict[str, object]] = []

        def advance(
            self,
            value: int,
            *,
            architecture: str,
            encoder: str,
            sample_index: int,
            total_samples: int,
            completed_trials: int,
            failed_trials: int,
        ) -> None:
            self.updated += value
            self.calls.append(
                {
                    "architecture": architecture,
                    "encoder": encoder,
                    "sample_index": sample_index,
                    "total_samples": total_samples,
                    "completed_trials": completed_trials,
                    "failed_trials": failed_trials,
                }
            )

        def finish(self) -> None:
            self.finished = True

    progress = DummyProgress()

    monkeypatch.setattr(
        "helpers.lr_finder.runner.prepare_training_data",
        lambda config, *, normalizer_device: data_bundle,
    )
    monkeypatch.setattr(
        "helpers.lr_finder.runner.sample_bcedice_params", lambda n, search_space, seed: samples
    )
    monkeypatch.setattr("helpers.lr_finder.runner.configure_execution_mode", lambda mode: None)
    monkeypatch.setattr("helpers.lr_finder.runner.seed_everything", lambda seed: None)
    monkeypatch.setattr("helpers.lr_finder.runner.GPUNormalizer", lambda **kwargs: lambda x: x)
    monkeypatch.setattr(
        "helpers.lr_finder.runner._SnapshotProgressReporter", lambda total_trials: progress
    )
    monkeypatch.setattr(
        "helpers.lr_finder.runner.GPUDownscale",
        lambda p: SimpleNamespace(to=lambda device: lambda x: x),
    )
    monkeypatch.setattr("helpers.lr_finder.runner.require_cuda_device", lambda: "cuda")

    def fake_capture_pretrained_model_state(
        model_plan: ModelPlan, device: object
    ) -> dict[str, torch.Tensor]:
        del device
        preload_calls.append((model_plan.architecture, model_plan.encoder))
        return {"weight": torch.tensor([1.0])}

    monkeypatch.setattr(
        "helpers.lr_finder.runner._capture_pretrained_model_state",
        fake_capture_pretrained_model_state,
    )

    def fake_run_single(*_args: object, **_kwargs: object) -> tuple[RunRecord | None, int, int]:
        call_count["value"] += 1
        if call_count["value"] == 1:
            return (
                RunRecord(
                    architecture="FPN",
                    encoder="resnet34",
                    alpha=0.1,
                    beta=0.2,
                    gamma=0.3,
                    median_min_loss=0.4,
                    plot_path=config.output_dir / "FPN" / "001.png",
                    csv_path=config.output_dir / "FPN" / "SUMMARY_FPN_STABILITY.csv",
                ),
                1,
                0,
            )
        return (None, 0, 1)

    monkeypatch.setattr("helpers.lr_finder.runner._run_single_loss_config", fake_run_single)

    outputs = run_lr_finder_screening(config)

    assert outputs.completed_trials == 1
    assert outputs.failed_trials == 1
    assert outputs.architecture_trial_stats == {
        "FPN": {"valid_records": 1, "completed_trials": 1, "failed_trials": 1}
    }
    assert outputs.summary_all_path.is_file()
    assert outputs.lhs_samples_path.is_file()
    assert outputs.architecture_summary_paths["FPN"].is_file()
    assert dataset.closed is True
    assert preload_calls == [("FPN", "resnet34")]
    assert progress.updated == len(samples) * config.num_repeats
    assert progress.finished is True
    assert progress.calls == [
        {
            "architecture": "FPN",
            "encoder": "resnet34",
            "sample_index": 1,
            "total_samples": 2,
            "completed_trials": 1,
            "failed_trials": 0,
        },
        {
            "architecture": "FPN",
            "encoder": "resnet34",
            "sample_index": 2,
            "total_samples": 2,
            "completed_trials": 1,
            "failed_trials": 1,
        },
    ]
