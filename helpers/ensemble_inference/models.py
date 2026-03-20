from __future__ import annotations

from typing import Any

import torch
from torch import nn

from helpers.ensemble_inference.recipe import EnsembleRecipe
from helpers.ensemble_optimizer.models import load_checkpoint_strict_without_aux
from helpers.training.models import create_model


def load_recipe_models(
    recipe: EnsembleRecipe,
    device: torch.device,
) -> tuple[list[nn.Module], list[dict[str, Any]]]:
    models: list[nn.Module] = []
    constituent_info: list[dict[str, Any]] = []
    for spec in recipe.model_registry:
        if not spec.checkpoint_path.exists():
            raise FileNotFoundError(f"Recipe checkpoint not found: {spec.checkpoint_path}")
        model = create_model(spec.architecture, spec.encoder, validation=True)
        model = load_checkpoint_strict_without_aux(model, str(spec.checkpoint_path), device)
        model.eval()
        models.append(model)
        constituent_info.append(dict(spec.raw_metadata))
    return models, constituent_info
