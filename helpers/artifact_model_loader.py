from __future__ import annotations

from dataclasses import dataclass

from helpers.artifact_config import ArtifactDetectionConfig


@dataclass(frozen=True)
class LoadedArtifactModels:
    """Holds the loaded tissue detector and QC model."""

    tissue_model: object
    preprocessing_fn: object
    qc_model: object


class ArtifactModelLoader:
    """Lazy loader for artifact detection models."""

    def __init__(self, config: ArtifactDetectionConfig) -> None:
        self.config = config
        self._cached_models: LoadedArtifactModels | None = None

    def load(self) -> LoadedArtifactModels:
        """Load the tissue detector and QC model once per run."""

        if self._cached_models is None:
            import os

            import segmentation_models_pytorch as smp
            import torch

            preprocessing_fn = smp.encoders.get_preprocessing_fn(
                self.config.encoder_model_td,
                self.config.encoder_weights_td,
            )
            tissue_model = smp.UnetPlusPlus(
                encoder_name=self.config.encoder_model_td,
                encoder_weights=self.config.encoder_weights_td,
                classes=2,
                activation=None,
            )
            tissue_model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.config.tissue_detector_model_dir,
                        self.config.tissue_detector_model_name,
                    ),
                    map_location="cpu",
                    weights_only=False,
                )
            )
            tissue_model.to(self.config.device)
            tissue_model.eval()

            qc_model = torch.load(
                self.config.qc_model_dir / self.config.qc_model_name,
                map_location=self.config.device,
                weights_only=False,
            )
            self._cached_models = LoadedArtifactModels(
                tissue_model=tissue_model,
                preprocessing_fn=preprocessing_fn,
                qc_model=qc_model,
            )
        return self._cached_models
