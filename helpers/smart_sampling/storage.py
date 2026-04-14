from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from helpers.patient_shard_cache import PatientShardCache
from helpers.smart_sampling.config import SmartSamplerConfig


@dataclass(frozen=True)
class PreparedSmartSamplerStorage:
    source_shard_dir: Path
    source_manifest_path: Path
    final_output_dir: Path
    sidecar_output_dir: Path
    patient_shard_cache: PatientShardCache | None
    should_publish_outputs: bool


def prepare_storage(config: SmartSamplerConfig) -> PreparedSmartSamplerStorage:
    needs_local_work_dir = config.stage_input_locally or config.stage_outputs_locally
    local_work_dir = _require_local_work_dir(config) if needs_local_work_dir else None

    source_shard_dir = _require_source_shard_dir(config)
    source_manifest_path = _require_source_manifest_path(config)
    final_output_dir = config.output_dir
    sidecar_output_dir = config.output_dir
    patient_shard_cache: PatientShardCache | None = None
    if needs_local_work_dir:
        assert local_work_dir is not None
        _input_stage_dir(local_work_dir).mkdir(parents=True, exist_ok=True)
        _output_stage_dir(local_work_dir).mkdir(parents=True, exist_ok=True)
    if config.stage_input_locally and config.patient_shard_cache_bytes > 0:
        cache_dir = config.patient_shard_cache_dir or _cache_stage_dir(
            _require_local_work_dir(config)
        )
        patient_shard_cache = PatientShardCache(
            cache_dir=cache_dir,
            size_cap_bytes=config.patient_shard_cache_bytes,
        )
    if config.stage_outputs_locally:
        assert local_work_dir is not None
        sidecar_output_dir = _output_stage_dir(local_work_dir)
        logging.info("Writing Stage 7 sidecars locally first in %s", sidecar_output_dir)

    return PreparedSmartSamplerStorage(
        source_shard_dir=source_shard_dir,
        source_manifest_path=source_manifest_path,
        final_output_dir=final_output_dir,
        sidecar_output_dir=sidecar_output_dir,
        patient_shard_cache=patient_shard_cache,
        should_publish_outputs=config.stage_outputs_locally,
    )


def publish_outputs(
    config: SmartSamplerConfig,
    *,
    selection_csv_path: Path | None,
    stats_csv_path: Path | None,
    run_config_path: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Publishing Stage 7 sidecars back to %s", config.output_dir)
    published_selection_csv_path = _publish_optional_file(selection_csv_path, config.output_dir)
    published_stats_csv_path = _publish_optional_file(stats_csv_path, config.output_dir)
    published_run_config_path = _publish_optional_file(run_config_path, config.output_dir)
    return (published_selection_csv_path, published_stats_csv_path, published_run_config_path)


def stage_patient_source_shard(
    config: SmartSamplerConfig,
    storage: PreparedSmartSamplerStorage,
    source_shard_path: Path,
) -> Path:
    if not config.stage_input_locally:
        return source_shard_path
    if storage.patient_shard_cache is not None:
        return storage.patient_shard_cache.fetch(source_shard_path)
    local_work_dir = _require_local_work_dir(config)
    staged_input_path = _input_stage_dir(local_work_dir) / source_shard_path.name
    logging.info("Staging source shard locally: %s -> %s", source_shard_path, staged_input_path)
    shutil.copy2(source_shard_path, staged_input_path)
    return staged_input_path


def prepare_patient_output_path(
    config: SmartSamplerConfig,
    *,
    patient_id: int,
) -> Path:
    base_dir = (
        _output_stage_dir(_require_local_work_dir(config))
        if config.stage_outputs_locally
        else config.output_dir
    )
    output_dir = base_dir / config.output_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{patient_id}.h5"


def published_patient_output_path(config: SmartSamplerConfig, *, patient_id: int) -> Path:
    return config.output_dir / config.output_filename / f"{patient_id}.h5"


def publish_patient_output(config: SmartSamplerConfig, local_output_path: Path) -> Path:
    destination_dir = config.output_dir / config.output_filename
    destination_dir.mkdir(parents=True, exist_ok=True)
    return _publish_path(local_output_path, destination_dir)


def cleanup_patient_workspace(
    *, transient_input_path: Path | None, transient_output_path: Path | None
) -> None:
    for path in (transient_input_path, transient_output_path):
        if path is not None and path.exists():
            path.unlink()


def cleanup_local_work_dir(config: SmartSamplerConfig) -> None:
    if (
        not config.clean_local_work_dir
        or config.local_work_dir is None
        or not (config.stage_input_locally or config.stage_outputs_locally)
    ):
        return
    if config.local_work_dir.exists():
        logging.info("Cleaning local smart-sampling work dir %s", config.local_work_dir)
        shutil.rmtree(config.local_work_dir)


def prepare_source_h5(config: SmartSamplerConfig) -> Path:
    if config.source_h5_path is None:
        raise ValueError(
            "SMART_SAMPLER_SOURCE_H5 is no longer supported for Stage 7 pipeline input."
        )
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


def _require_source_shard_dir(config: SmartSamplerConfig) -> Path:
    if config.source_shard_dir is None:
        raise ValueError("SMART_SAMPLER_SOURCE_SHARD_DIR is required for Stage 7 shard input.")
    return config.source_shard_dir


def _require_source_manifest_path(config: SmartSamplerConfig) -> Path:
    if config.source_manifest_path is None:
        raise ValueError("SMART_SAMPLER_SOURCE_MANIFEST_PATH is required for Stage 7 shard input.")
    return config.source_manifest_path


def _input_stage_dir(local_work_dir: Path) -> Path:
    return local_work_dir / "input"


def _output_stage_dir(local_work_dir: Path) -> Path:
    return local_work_dir / "output"


def _cache_stage_dir(local_work_dir: Path) -> Path:
    return local_work_dir / "cache" / "patient_shards"


def _publish_optional_file(path: Path | None, destination_dir: Path) -> Path | None:
    if path is None:
        return None
    return _publish_path(path, destination_dir)


def _publish_path(source_path: Path, destination_dir: Path) -> Path:
    destination_path = destination_dir / source_path.name
    temporary_destination_path = destination_path.with_name(f".{destination_path.name}.tmp")
    if temporary_destination_path.exists():
        if temporary_destination_path.is_dir():
            shutil.rmtree(temporary_destination_path)
        else:
            temporary_destination_path.unlink()
    if source_path.is_dir():
        shutil.copytree(source_path, temporary_destination_path)
        if destination_path.exists():
            shutil.rmtree(destination_path)
    else:
        shutil.copy2(source_path, temporary_destination_path)
    temporary_destination_path.replace(destination_path)
    return destination_path
