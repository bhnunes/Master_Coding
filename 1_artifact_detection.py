from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from helpers.artifact.config import ArtifactDetectionConfig, load_artifact_detection_config
from helpers.artifact.logging import Style, banner, configure_artifact_logger
from helpers.artifact.model_loader import ArtifactModelLoader
from helpers.artifact.pipeline import ArtifactDetectionPipeline, ArtifactLogger
from helpers.artifact.processor import ArtifactProcessor
from helpers.artifact.repository import ArtifactRepository
from helpers.artifact.zip import ZipSlideSource


def main() -> None:
    """Run Stage 1 artifact detection from `.env` configuration."""

    load_dotenv(override=True)
    config = load_artifact_detection_config(os.environ)

    logger: ArtifactLogger = configure_artifact_logger(config.log_path)
    logger.info(banner(f"{Style.ROCKET} STAGE 1 ARTIFACT DETECTION"))
    _ensure_runtime_directories(
        config.database_folder, config.geojson_output, config.temp_root, config.log_folder
    )
    _log_configuration(logger, config)

    repository = ArtifactRepository(config.database_path)
    zip_source = ZipSlideSource(config.images_zip)
    model_loader = ArtifactModelLoader(config)
    processor = ArtifactProcessor(config, model_loader)
    pipeline = ArtifactDetectionPipeline(
        zip_source=zip_source,
        repository=repository,
        processor=processor,
        temp_root=config.temp_root,
        geojson_output=config.geojson_output,
        logger=logger,
    )

    summary = pipeline.run()
    logger.info(banner(f"{Style.SUCCESS} ARTIFACT DETECTION COMPLETE"))
    logger.info(
        "%s Discovered=%s | Processed=%s | Failed=%s",
        Style.INFO,
        summary.discovered,
        summary.processed,
        summary.failed,
    )
    logger.info("%s Database: %s", Style.DB, config.database_path)
    logger.info("%s GeoJSON output: %s", Style.FOLDER, config.geojson_output)


def _ensure_runtime_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def _log_configuration(logger: ArtifactLogger, config: ArtifactDetectionConfig) -> None:
    logger.info("%s Zip source: %s", Style.ZIP, config.images_zip)
    logger.info("%s Database folder: %s", Style.DB, config.database_folder)
    logger.info("%s Temp folder: %s", Style.FOLDER, config.temp_root)
    logger.info("%s GeoJSON output: %s", Style.FOLDER, config.geojson_output)
    logger.info("%s Device: %s", Style.IMAGE, config.device)


if __name__ == "__main__":
    main()
