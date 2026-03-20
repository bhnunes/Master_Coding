from __future__ import annotations

import shutil
from pathlib import Path

from helpers.smart_sampling.config import SmartSamplerConfig


def prepare_source_h5(config: SmartSamplerConfig) -> Path:
    if not config.stage_input_locally:
        return config.source_h5_path
    if config.local_work_dir is None:
        raise ValueError("SMART_SAMPLER_LOCAL_WORK_DIR is required when local staging is enabled.")
    config.local_work_dir.mkdir(parents=True, exist_ok=True)
    staged_path = config.local_work_dir / config.source_h5_path.name
    shutil.copy2(config.source_h5_path, staged_path)
    return staged_path
