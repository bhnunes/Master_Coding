# Method Sources

This file maps major manuscript sections in `reports/main.tex` to repository sources inspected during drafting.
The repo now uses current script names (`3_1`, `4_crossfold`, `8_training_ensemble`, etc.),
so stage labels here follow the current codebase naming used in orchestration entrypoints.

## Pipeline overview

- `README.md`
- `AGENTS.md`
- `1_artifact_detection.py`
- `2_database_manager.py`
- `3_1_optimization_sampling.py`
- `3_2_tune_graph_method.py`
- `3_3_cleaner_script.py`
- `4_crossfold.py`
- `5_sanity_checks.py`
- `6_smart_sampler.py`
- `7_lr_finder.py`
- `8_training_ensemble.py`
- `9_optimizer_ensemble.py`
- `10_inference_ensemble.py`
- `helpers/runtime_platform.py`
- `helpers/runtime_normalization.py`
- `helpers/logging_utils.py`

## Stage 1 artifact detection

- `1_artifact_detection.py`
- `helpers/artifact/config.py`
- `helpers/artifact/pipeline.py`
- `helpers/artifact/processor.py`
- `helpers/artifact/model_loader.py`
- `helpers/artifact/repository.py`
- `helpers/artifact/zip.py`
- `.env_example`

## Stage 2 database-backed patch extraction

- `2_database_manager.py`
- `helpers/extraction/config.py`
- `helpers/extraction/repository.py`
- `helpers/extraction/image_reader_service.py`
- `helpers/extraction/data_handlers.py`
- `helpers/extraction/master_manifest.py`
- `helpers/extraction/patch_engine.py`
- `helpers/runtime_platform.py`
- `.env_example`

## Stages 3.1--3.3 cleaning workflow

- `3_1_optimization_sampling.py`
- `helpers/optimization_sampling/config.py`
- `helpers/optimization_sampling/sampling.py`
- `helpers/optimization_sampling/overlay.py`
- `helpers/optimization_sampling/pipeline.py`
- `3_2_tune_graph_method.py`
- `helpers/graph/tuning_config.py`
- `helpers/graph/tuning_pipeline.py`
- `helpers/graph/contamination.py`
- `helpers/graph/parameter_store.py`
- `3_3_cleaner_script.py`
- `helpers/graph/cleaning_config.py`
- `helpers/graph/cleaning_pipeline.py`
- `.env_example`

## Stage 3 canonical source packaging helpers

- `helpers/packaging/config.py`
- `helpers/packaging/pipeline.py`
- `helpers/packaging/writer.py`
- `.env_example`

## Stage 4 crossfold and normalization

- `4_crossfold.py`
- `helpers/crossfold/config.py`
- `helpers/crossfold/discovery.py`
- `helpers/crossfold/entropy.py`
- `helpers/crossfold/io.py`
- `helpers/crossfold/normalization.py`
- `helpers/crossfold/pipeline.py`
- `helpers/crossfold/provenance.py`
- `helpers/crossfold/splitting.py`
- `helpers/runtime_normalization.py`
- `.env_example`

## Stage 5 sanity checks

- `5_sanity_checks.py`
- `helpers/sanity/config.py`
- `helpers/sanity/contracts.py`
- `helpers/sanity/disk_checks.py`
- `helpers/sanity/manifest_checks.py`
- `helpers/sanity/pipeline.py`
- `helpers/sanity/provenance.py`
- `helpers/sanity/reporting.py`
- `helpers/sanity/semantic_checks.py`

## Stage 6 smart sampling

- `6_smart_sampler.py`
- `helpers/smart_sampling/config.py`
- `helpers/smart_sampling/index.py`
- `helpers/smart_sampling/embeddings.py`
- `helpers/smart_sampling/gist.py`
- `helpers/smart_sampling/selection.py`
- `helpers/smart_sampling/storage.py`
- `helpers/smart_sampling/writer.py`
- `helpers/smart_sampling/pipeline.py`
- `helpers/training/compact_train_selected.py`

## Stage 7 learning-rate screening

- `7_lr_finder.py`
- `helpers/lr_finder/config.py`
- `helpers/lr_finder/data.py`
- `helpers/lr_finder/search_space.py`
- `helpers/lr_finder/analysis.py`
- `helpers/lr_finder/runner.py`
- `helpers/lr_finder/reporting.py`
- `helpers/lr_finder/pipeline.py`
- `helpers/runtime_normalization.py`
- `helpers/training/stain_normalization.py`
- `helpers/training/compact_train_selected.py`
- `training_model_registry_NOT_NORMALIZED.json`
- `training_model_registry_MACENKO.json`
- `training_model_registry_REINHARD.json`
- `training_model_registry_RUIFROK.json`
- `training_model_registry_VAHADANE.json`

## Stage 8 training

- `8_training_ensemble.py`
- `helpers/training/config.py`
- `helpers/training/data.py`
- `helpers/training/compact_train_selected.py`
- `helpers/training/models.py`
- `helpers/training/losses.py`
- `helpers/training/loop.py`
- `helpers/training/metrics.py`
- `helpers/training/checkpointing.py`
- `helpers/training/pipeline.py`
- `helpers/training/runtime.py`
- `helpers/training/gpu.py`
- `helpers/runtime_normalization.py`
- `helpers/training/stain_normalization.py`
- `training_model_registry_NOT_NORMALIZED.json`
- `training_model_registry_MACENKO.json`
- `training_model_registry_REINHARD.json`
- `training_model_registry_RUIFROK.json`
- `training_model_registry_VAHADANE.json`

## Stages 9--10 ensemble optimization and inference

- `9_optimizer_ensemble.py`
- `helpers/ensemble_optimizer/config.py`
- `helpers/ensemble_optimizer/data.py`
- `helpers/ensemble_optimizer/metadata.py`
- `helpers/ensemble_optimizer/models.py`
- `helpers/ensemble_optimizer/optimization.py`
- `helpers/ensemble_optimizer/pipeline.py`
- `helpers/ensemble_optimizer/reporting.py`
- `helpers/ensemble_optimizer/splitting.py`
- `helpers/runtime_normalization.py`
- `10_inference_ensemble.py`
- `helpers/ensemble_inference/config.py`
- `helpers/ensemble_inference/data.py`
- `helpers/ensemble_inference/inference.py`
- `helpers/ensemble_inference/metrics.py`
- `helpers/ensemble_inference/models.py`
- `helpers/ensemble_inference/pipeline.py`
- `helpers/ensemble_inference/recipe.py`
- `helpers/ensemble_inference/reporting.py`
- `helpers/runtime_normalization.py`

## Environment and template constraints

- `pyproject.toml`
- `.env_example`
- `latex_template/README`
- `latex_template/cas-dc-template.tex`
- `latex_template/cas-dc-sample.tex`
