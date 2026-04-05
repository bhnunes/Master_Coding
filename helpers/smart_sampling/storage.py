from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from helpers.smart_sampling.config import SmartSamplerConfig


@dataclass(frozen=True)
class PreparedSmartSamplerStorage:
    source_h5_path: Path
    output_dir: Path
    should_publish_outputs: bool


def prepare_storage(config: SmartSamplerConfig) -> PreparedSmartSamplerStorage:
    needs_local_work_dir = config.stage_input_locally or config.stage_outputs_locally
    local_work_dir = _require_local_work_dir(config) if needs_local_work_dir else None

    source_h5_path = config.source_h5_path
    if config.stage_input_locally:
        assert local_work_dir is not None
        staged_input_dir = _input_stage_dir(local_work_dir)
        staged_input_dir.mkdir(parents=True, exist_ok=True)
        staged_path = staged_input_dir / config.source_h5_path.name
        shutil.copy2(config.source_h5_path, staged_path)
        source_h5_path = staged_path

    output_dir = config.output_dir
    if config.stage_outputs_locally:
        assert local_work_dir is not None
        output_dir = _output_stage_dir(local_work_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    return PreparedSmartSamplerStorage(
        source_h5_path=source_h5_path,
        output_dir=output_dir,
        should_publish_outputs=config.stage_outputs_locally,
    )


def publish_outputs(
    config: SmartSamplerConfig,
    *,
    filtered_h5_path: Path,
    selection_csv_path: Path | None,
    stats_csv_path: Path | None,
    run_config_path: Path | None,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    published_filtered_h5_path = _publish_file(filtered_h5_path, config.output_dir)
    published_selection_csv_path = _publish_optional_file(selection_csv_path, config.output_dir)
    published_stats_csv_path = _publish_optional_file(stats_csv_path, config.output_dir)
    published_run_config_path = _publish_optional_file(run_config_path, config.output_dir)
    return (
        published_filtered_h5_path,
        published_selection_csv_path,
        published_stats_csv_path,
        published_run_config_path,
    )


def cleanup_local_work_dir(config: SmartSamplerConfig) -> None:
    if (
        not config.clean_local_work_dir
        or config.local_work_dir is None
        or not (config.stage_input_locally or config.stage_outputs_locally)
    ):
        return
    if config.local_work_dir.exists():
        shutil.rmtree(config.local_work_dir)


def prepare_source_h5(config: SmartSamplerConfig) -> Path:
    if not config.stage_input_locally:
        return config.source_h5_path
    local_work_dir = _require_local_work_dir(config)
    staged_input_dir = _input_stage_dir(local_work_dir)
    staged_input_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staged_input_dir / config.source_h5_path.name
    shutil.copy2(config.source_h5_path, staged_path)
    return staged_path


def _require_local_work_dir(config: SmartSamplerConfig) -> Path:
    if config.local_work_dir is None:
        raise ValueError("SMART_SAMPLER_LOCAL_WORK_DIR is required when local staging is enabled.")
    return config.local_work_dir


def _input_stage_dir(local_work_dir: Path) -> Path:
    return local_work_dir / "input"


def _output_stage_dir(local_work_dir: Path) -> Path:
    return local_work_dir / "output"


def _publish_optional_file(path: Path | None, destination_dir: Path) -> Path | None:
    if path is None:
        return None
    return _publish_file(path, destination_dir)


def _publish_file(source_path: Path, destination_dir: Path) -> Path:
    destination_path = destination_dir / source_path.name
    temporary_destination_path = destination_path.with_name(f".{destination_path.name}.tmp")
    shutil.copy2(source_path, temporary_destination_path)
    temporary_destination_path.replace(destination_path)
    return destination_path
