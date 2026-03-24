=========================================================================================

# AGENTS.md

This file gives coding agents repository-specific guidance for working safely and effectively in this codebase.

## Purpose

- This repository is a Python research pipeline for pathology whole-slide-image processing.
- This is a script-first research pipeline.
- Main responsibilities include patch extraction, dataset splitting, scientific sanity checks, and HDF5 packaging.
- The repo is script-driven rather than package-driven.

## Repository Layout
- `1_artifact_detection.py`: Generation of GeoJSON information about artifacts on the whole-slide-images.
- `2_database_manager.py`: `.env`-driven Stage 2 orchestrator for ingestion, SQLite tracking, project setup, per-case patch extraction, and optional per-slide local SSD staging from external WSI storage.
- `3_1_imageReader.py`: per-slide worker CLI; selects the correct annotation handler and runs extraction.
- `helpers/extraction/data_handlers.py`: annotation format adapters for `.svs/.xml`, `.ndpi/.ndpa`, and JSON-based formats.
- `helpers/extraction/patch_engine.py`: main patch extraction engine; tissue checks, polygon masking, artifact filtering, image/mask writes.
- `4_1_optimization_sampling.py`: Script responsible for selecting a subsample of cancer images for a human-in-the-loop cleaning process of incorrect annotations.
- `4_2_tune_graph_method.py`: Script responsible for detecting the best parameters to be used on a graph segmentation method that will be used to remove incorrectly annotated images.
- `4_3_cleaner_script.py`: responsible for applying the graph segmentation method with the parameters obtained to remove incorrectly annotated images from cancer folder.
- `5_pack_splits_to_hdf5.py`: packages the cleaned post-Stage-4.3 patch pool into one source HDF5 dataset consumed by Stage 6.
- `6_crossfold.py`: patient-level dataset split creation plus optional stain normalization, operating on HDF5 patch datasets.
- `7_sanity_checks.py`: scientific integrity and dataset consistency checks for HDF5-based split artifacts.
- `8_smart_sampler.py`: `.env`-driven smart-sampling orchestrator that filters `TRAIN.h5` into `TRAIN_FILTERED.h5` for downstream LR finding and training.
- `9_lr_finder.py`: `.env`-driven LR-finder orchestrator that screens approved model pairs and writes a LaTeX PDF report.
- `10_training_ensemble.py`: Thin Stage 8 training orchestrator that trains one approved model per execution.
- `11_optimizer_ensemble.py`: `.env`-driven Stage 11 orchestrator that optimizes the two-stream ensemble recipe used by Stage 12 inference.
- `12_inference_ensemble.py`: Responsible for generating the results on the test set.

## Architecture

- Each root script was created as a standalone application.
- The architecture is a monolith with Domain Driven Design.
- All environment variables should be saved in the `.env` file. This file must have comments indicating which script each variable belongs to.
- All test scripts must be saved in `/tests`.
- All logs generated must be saved in `/logs`.
- All helpers should be standalone modules or classes organized by domain and saved in `/helpers`.
- Domain-specific helpers must live inside `helpers/<domain>/` packages. If a domain package already exists, add new helper modules there instead of creating new flat modules directly under `helpers/`.
- Only broadly shared cross-domain utilities should remain top-level in `helpers/`, such as `helpers/runtime_platform.py` and `helpers/logging_utils.py`.
- Databases should be saved in `/databases`.

## Current Repository Notes

- `1_artifact_detection.py` is now a `.env`-driven Stage 1 orchestrator. Keep orchestration there and keep runtime/domain behavior in `helpers/artifact/*.py`.
- Stage 1 artifact detection reads WSI members directly from a zip archive, extracts one slide at a time to a temporary workspace, writes GeoJSON outputs, updates SQLite status, and cleans temporary files after each slide.
- Stage 1 GeoJSON outputs are intended to be reused by Stage 2 through `GEOJSON_PATH` when advanced artifact filtering is enabled.
- Stage 1 database tracking must preserve at least `Image_Name`, `GeoJSON_Processed`, and `Comments`. Operational metadata such as `LastUpdate` is also allowed.
- Stage 1 logging is intentionally colorful and user-facing. Preserve the styled console + file logging pattern established in `helpers/artifact/logging.py` and inspired by `2_database_manager.py`.
- Active Stage 1 helper modules are `helpers/artifact/config.py`, `helpers/artifact/logging.py`, `helpers/artifact/model_loader.py`, `helpers/artifact/paths.py`, `helpers/artifact/pipeline.py`, `helpers/artifact/processor.py`, `helpers/artifact/repository.py`, and `helpers/artifact/zip.py`.
- `2_database_manager.py` is now a `.env`-driven Stage 2 orchestrator. Keep orchestration there and keep configuration, repository access, and slide-processing services in `helpers/extraction/config.py`, `helpers/extraction/repository.py`, and `helpers/extraction/image_reader_service.py`.
- Stage 2 supports two execution modes: ingestion mode with `LOADCASES=True`, and processing mode that consumes pending database cases and dispatches per-slide extraction.
- Stage 2 initializes project folders and SQLite schema, validates optional GeoJSON sanity checks during ingestion, and records per-case extraction metadata such as status, comments, patch counts, runtime, and extraction parameters.
- Stage 2 can attach artifact GeoJSON files generated by Stage 1 when `USE_ADVANCED_ARTIFACT_FILTERING=True` and `GEOJSON_PATH` points to the Stage 1 output folder.
- Stage 2 is the correct place to support external-SSD WSI storage plus per-slide local SSD staging. When this mode is enabled, preserve ingestion and provenance against the original source paths, copy only one WSI at a time into a local temporary workspace, process it there, and clean the local staged copy in a fail-safe `finally` path.
- Stage 2 no longer drops patches using `artifact_policy.yaml` thresholds. Instead, it records per-patch artifact coverage metadata into a Parquet sidecar for downstream training.
- Stage 2 should preserve the filename-keyed artifact metadata contract written to `PATCHES/artifact_patch_index.parquet` with columns `filename`, `label`, `patient_id`, `slide_id`, `cov_fold`, `cov_penmarking`, `cov_oof`, `cov_darkspot_foreign`, and `cov_edge_airbubble`.
- When `USE_ADVANCED_ARTIFACT_FILTERING=False`, Stage 2 should still write the same Parquet schema with zero-valued coverage columns so downstream contracts remain stable.
- Recent Stage 2 CPU optimization work in `helpers/extraction/patch_engine.py` delivered two measured wins on representative slides under `PROJECTS_BASE_PATH`: vectorized candidate filtering via `shapely.contains_xy(...)` and faster per-window polygon clipping via `shapely.ops.clip_by_rect(...)`.
- The attempted dense raster/integral-image redesign for Stage 2 was benchmarked on representative CHILE slides and was slower at real slide scale; do not resume that approach as the default optimization direction unless a new benchmark first shows a clear win on real data.
- The latest representative Stage 2 full-slide profiling shows candidate filtering is no longer the main bottleneck. `read_region` is now the dominant cost, followed by image save and then tissue/mask work. If resuming Stage 2 performance work, start with `helpers/extraction/patch_engine.py` and target `read_region` / patch I/O strategy first.
- The current safe Stage 2 optimization baseline is: vectorized center-point candidate filtering, `clip_by_rect(...)`-based polygon clipping, overlap short-circuit before `read_region(...)`, delayed artifact coverage until after overlap+tissue acceptance, and streamed worker-result reduction instead of materializing the full result list.
- When resuming Stage 2 performance work, benchmark before and after on representative real samples from `PROJECTS_BASE_PATH` in `.env`, ideally at least one artifact-bearing slide and one larger non-GeoJSON slide. Preserve patch counts, labels, filenames, and artifact sidecar semantics while optimizing.
- Active Stage 2 helper modules are `helpers/extraction/config.py`, `helpers/extraction/repository.py`, `helpers/extraction/artifact_index.py`, `helpers/extraction/image_reader_service.py`, `helpers/extraction/data_handlers.py`, and `helpers/extraction/patch_engine.py`.
- `4_1_optimization_sampling.py` should stay orchestration-focused. Keep sample-size calculation, candidate ranking, overlay generation, and logging behavior in `helpers/optimization_sampling/config.py`, `helpers/optimization_sampling/sampling.py`, `helpers/optimization_sampling/overlay.py`, `helpers/optimization_sampling/pipeline.py`, and `helpers/optimization_sampling/logging.py`.
- `helpers/optimization_sampling/logging.py` is now a thin domain wrapper over shared stage-logger infrastructure in `helpers/logging_utils.py`. Put generic stage logger behavior in `helpers/logging_utils.py`, not back into optimization-sampling-specific modules.
- `5_pack_splits_to_hdf5.py` should stay orchestration-focused. Keep Stage 5 environment loading, cleaned-patch-pool discovery, pairing checks, and HDF5 writing in `helpers/packaging/*.py`.
- Stage 5 now sits logically after Stage 4.3 and before Stage 6. Its responsibility is to package the cleaned accepted patch pool into one single source HDF5 dataset, not to package pre-made split folders.
- The Stage 5 source HDF5 must preserve a stable row-wise contract centered on `images`, `masks`, `labels`, `patient_ids`, and `filenames`. Additional provenance metadata is allowed, but do not break the downstream filename-keyed joins or row alignment assumptions required by Stage 6, Stage 8, and artifact-aware training.
- Active Stage 5 helper modules are `helpers/packaging/config.py`, `helpers/packaging/discovery.py`, `helpers/packaging/pipeline.py`, and `helpers/packaging/writer.py`.
- `6_crossfold.py` should stay orchestration-focused. Keep Stage 6 environment loading, HDF5 dataset discovery, entropy scoring, split search, stain normalization, split writing, logging, and provenance behavior in `helpers/crossfold/*.py`.
- Stage 6 now targets an HDF5-native workflow with no PNG backward-compatibility path. It should read one cleaned source HDF5 dataset produced after Stage 4.3, create patient-disjoint `TRAIN`/`VALIDATION`/`TEST` splits, fit stain normalization parameters on `TRAIN` only, apply that frozen normalizer to `TRAIN`/`VALIDATION`/`TEST`, and write final split HDF5 artifacts for downstream stages.
- Stage 6 must preserve patient stratification by `max(patch_label)`, deterministic `filename` identity, exact image/mask row alignment, and the current provenance artifacts (`manifest.csv`, `split_stats.csv`, `run_config.json`, optional entropy caches, and normalization stats/templates when enabled), even though provenance will now reference HDF5 artifacts and row identity instead of PNG-relative paths.
- Active Stage 6 helper modules are `helpers/crossfold/config.py`, `helpers/crossfold/discovery.py`, `helpers/crossfold/entropy.py`, `helpers/crossfold/io.py`, `helpers/crossfold/logging.py`, `helpers/crossfold/normalization.py`, `helpers/crossfold/pipeline.py`, `helpers/crossfold/provenance.py`, and `helpers/crossfold/splitting.py`.
- `7_sanity_checks.py` should stay orchestration-focused. Keep Stage 7 configuration, provenance loading, manifest checks, HDF5 contract checks, semantic mask checks, reporting, and orchestration in `helpers/sanity/*.py`.
- Stage 7 is now HDF5-only with no PNG backward-compatibility path. It must preserve the reviewer-grade scientific validation goal under the HDF5-native split workflow: detect patient leakage, manifest/provenance drift, filename contract violations, HDF5 dataset parity issues, unreadable or shape-mismatched image/mask rows, invalid mask values, and label-mask semantic contradictions before Stage 8 training.
- Active Stage 7 helper modules are `helpers/sanity/config.py`, `helpers/sanity/contracts.py`, `helpers/sanity/disk_checks.py`, `helpers/sanity/manifest_checks.py`, `helpers/sanity/models.py`, `helpers/sanity/pipeline.py`, `helpers/sanity/provenance.py`, `helpers/sanity/reporting.py`, and `helpers/sanity/semantic_checks.py`.
- `8_smart_sampler.py` should stay orchestration-focused. Keep Stage 8 smart-sampling configuration, HDF5 indexing, embedding extraction, per-patient selection, output writing, and optional sidecar generation in `helpers/smart_sampling/*.py`.
- Stage 8 smart sampling must preserve downstream `TRAIN_FILTERED.h5` compatibility for `9_lr_finder.py` and `10_training_ensemble.py`, always writing datasets named exactly `images`, `masks`, `labels`, `patient_ids`, and `filenames`.
- Active Stage 8 smart-sampling helper modules are `helpers/smart_sampling/config.py`, `helpers/smart_sampling/index.py`, `helpers/smart_sampling/embeddings.py`, `helpers/smart_sampling/selection.py`, `helpers/smart_sampling/storage.py`, `helpers/smart_sampling/writer.py`, and `helpers/smart_sampling/pipeline.py`.
- `9_lr_finder.py` should stay orchestration-focused. Keep Stage 9 configuration, HDF5 staging, repeated LR-finder screening, curve analysis, and LaTeX/PDF reporting in `helpers/lr_finder/*.py`.
- Stage 9 must preserve compatibility with Stage 8 smart-sampling outputs, reuse the approved architecture/encoder matrix from `training_model_registry.json`, and always emit both `report.tex` and `report.pdf` plus reproducibility sidecars.
- Active Stage 9 helper modules are `helpers/lr_finder/config.py`, `helpers/lr_finder/data.py`, `helpers/lr_finder/search_space.py`, `helpers/lr_finder/analysis.py`, `helpers/lr_finder/runner.py`, `helpers/lr_finder/reporting.py`, and `helpers/lr_finder/pipeline.py`.
- `10_training_ensemble.py` is now a thin Stage 8 orchestrator. Keep orchestration there and keep training configuration, registry loading, data access, GPU utilities, runtime setup, losses, metrics, checkpointing, reporting, and epoch execution in `helpers/training/*.py`.
- `11_optimizer_ensemble.py` is now a thin Stage 11 orchestrator. Keep orchestration there and keep configuration, metadata selection, validation-data staging, model loading, prediction caching, split logic, Optuna objectives, holdout evaluation, calibration, and JSON reporting in `helpers/ensemble_optimizer/*.py`.
- Stage 11 is a development/tuning stage, not a claim-bearing evaluation stage. Treat Stage 11 holdout or calibration metrics as development-only unless the user explicitly asks for a stricter publication-grade redesign.
- Stage 11 must preserve the declarative Stage 12 recipe contract with `ensemble_strategy="two_stream_spatial_gating"` plus stable `roi_config`, `decision_config`, `spatial_config`, `model_registry`, and `holdout_metrics` fields.
- Stage 11 candidate loading must enforce provenance compatibility before comparing or mixing models. Do not silently optimize across checkpoints trained from incompatible split lineage, packaging lineage, normalization lineage, smart-sampling lineage, or artifact-aware-loss lineage.
- Stage 11 training metadata should carry a canonical provenance payload plus a deterministic compatibility signature so candidate filtering can fail closed when incompatible models are present.
- Preferred Stage 11 scientific design is a 3-way patient-disjoint split inside `VALIDATION`: optimization subset, calibration subset, and optional development holdout subset.
- When implementing the Stage 11 redesign, tune `roi_gate_threshold` during recipe optimization and tune a separate final `decision_threshold` on the internal calibration subset using `MCC`, then freeze both before Stage 12.
- Because Stage 11 now needs optimization/calibration/holdout patient subsets, Stage 5 should be treated as downstream-aware: validation sizing must be large enough for scientifically representative Stage 11 subsets.
- Scientific sizing guidance for the new Stage 11 design: bare minimum is roughly 12 validation patients total if classes are balanced, minimum defensible target is about 30 validation patients total, and recommended target is roughly 48-60 validation patients total; the real constraint is per-class patient counts, ideally at least 15 positive and 15 negative validation patients when the dataset can support it.
- Active Stage 11 helper modules are `helpers/ensemble_optimizer/config.py`, `helpers/ensemble_optimizer/data.py`, `helpers/ensemble_optimizer/metadata.py`, `helpers/ensemble_optimizer/models.py`, `helpers/ensemble_optimizer/optimization.py`, `helpers/ensemble_optimizer/reporting.py`, `helpers/ensemble_optimizer/splitting.py`, and `helpers/ensemble_optimizer/pipeline.py`.
- `12_inference_ensemble.py` is now a thin Stage 12 orchestrator. Keep orchestration there and keep configuration, recipe parsing, test-data staging, recipe-model loading, two-stream inference, metrics, and reporting in `helpers/ensemble_inference/*.py`.
- Stage 12 is the claim-bearing evaluation endpoint for final generalization results and should be the first stage where the frozen ensemble recipe is evaluated on `TEST`.
- Stage 12 must preserve compatibility with the Stage 11 recipe contract, always enforcing `ensemble_strategy="two_stream_spatial_gating"` and preserving `roi_config`, `decision_config`, `model_registry`, `holdout_metrics`, patient-level evaluation, confusion-matrix export, CSV export, and LaTeX/PDF report generation.
- Stage 12 must treat ROI gating and final hard-decision thresholding as separate semantics. Do not reuse a single threshold for both operations.
- Active Stage 12 helper modules are `helpers/ensemble_inference/config.py`, `helpers/ensemble_inference/recipe.py`, `helpers/ensemble_inference/data.py`, `helpers/ensemble_inference/models.py`, `helpers/ensemble_inference/inference.py`, `helpers/ensemble_inference/metrics.py`, `helpers/ensemble_inference/reporting.py`, and `helpers/ensemble_inference/pipeline.py`.
- `4_2_tune_graph_method.py` should be a `.env`-driven Stage 4.2 orchestrator. Keep orchestration there and keep configuration, shared contamination logic, and tuning workflow in `helpers/graph/tuning_config.py`, `helpers/graph/contamination.py`, and `helpers/graph/tuning_pipeline.py`.
- Stage 4.2 must preserve the current scientific workflow: human labels from `master_candidate_pool/APPROVED` and `REJECTED`, source image/mask pairing by filename stem, stratified train/test split, nested cross-validation for parameter search, F1 optimization on the `Rejected` class, and final held-out test evaluation.
- Stage 4.2 graph contamination logic is now shared domain logic. Future edits must avoid re-implementing the ROI contamination metric in root scripts; reuse `helpers/graph/contamination.py` so tuning and cleaning remain aligned.
- Stage 4.2 and Stage 4.3 share tuned-parameter persistence through `helpers/graph/parameter_store.py`; keep the on-disk parameter contract aligned between tuning and cleaning.
- `4_3_cleaner_script.py` should be a `.env`-driven Stage 4.3 orchestrator. Keep orchestration there and keep configuration and filtering workflow in `helpers/graph/cleaning_config.py` and `helpers/graph/cleaning_pipeline.py`.
- Stage 4.3 must reuse `helpers/graph/contamination.py` for the ROI contamination metric and load the tuned graph parameters plus `tau` through typed config so Stage 4.2 and Stage 4.3 stay scientifically aligned.
- Stage 4.3 must preserve the current accepted/rejected move semantics: accepted files remain in place, rejected images move to `REJECTED_IMAGES`, rejected masks move to `REJECTED_MASKS`, and missing/invalid pairs are skipped with explicit logging under `/logs`.
- The current planned migration boundary is: keep PNG-based patch handling through Stage 4.3, then convert the cleaned accepted patch pool into one single source HDF5 dataset in Stage 5, and keep Stage 6 and Stage 7 fully HDF5-native after that boundary.
- Stage 8 trains one model per execution, not the entire ensemble in a single run.
- Stage 8 architecture/encoder pairs are intentionally restricted to the approved research matrix in `.env_example` and `training_model_registry.json`; do not expand support casually.
- Stage 8 learning-rate and weight-decay defaults come from `training_model_registry.json`, optionally overridden with `TRAINING_MODEL_REGISTRY_PATH`.
- Stage 8 can optionally load `TRAINING_ARTIFACT_INDEX_PATH` and apply artifact-aware loss discounting keyed by HDF5 `filenames`.
- Active Stage 8 helper modules are `helpers/training/config.py`, `helpers/training/registry.py`, `helpers/training/models.py`, `helpers/training/losses.py`, `helpers/training/checkpointing.py`, `helpers/training/metrics.py`, `helpers/training/data.py`, `helpers/training/gpu.py`, `helpers/training/utils.py`, `helpers/training/loop.py`, `helpers/training/reporting.py`, `helpers/training/pipeline.py`, and `helpers/training/runtime.py`.
- Active WSI/extraction support helpers include `helpers/extraction/data_handlers.py`, `helpers/extraction/patch_engine.py`, `helpers/wsi/colors.py`, `helpers/wsi/maps.py`, `helpers/wsi/process.py`, `helpers/wsi/slide_info.py`, and `helpers/wsi/tis_detect_helper_fx.py`.
- Removed legacy helper files that should not be reintroduced without clear need: `helpers/main.py`, `helpers/wsi_tis_detect.py`, and `helpers/wsi_stain_norm.py`.
- Recent cleanup removed stale Stage 5/6 PNG compatibility code and dead checksum-only Stage 6 paths; do not reintroduce those legacy branches unless the user explicitly requests a new migration layer.
- Use package-safe imports from `helpers...` for helper modules. Do not add new sibling-style imports such as `from data_handlers import ...`.
- `pyproject.toml` is the only dependency source of truth for Python dependencies.
- `helpers/runtime_platform.py` centralizes runtime and path behavior across Linux, native Windows, WSL, containers, and Colab. Reuse it instead of open-coding platform checks.
- During future refactors, follow the same packaging pattern: if behavior belongs to a specific domain, create or reuse `helpers/<domain>/` and place the module there. Do not add new top-level helper modules for domain logic when an existing domain package already fits.
- `setup_colab.sh` is the canonical Google Colab bootstrap. It installs system dependencies, installs `uv`, installs Python 3.12, syncs from `pyproject.toml`, and prepares `.env` when missing.
- `setup_windows.ps1` is the canonical native Windows bootstrap. `Dockerfile` and `.devcontainer/devcontainer.json` define the containerized development environments.
- Current validation status after the Stage 5/6 HDF5 cleanup pass: targeted packaging/crossfold/sanity migration tests passed (`49 passed`), additional cleanup-focused tests passed (`72 passed` and `9 passed` on targeted subsets), `uv run ruff check . --select ARG001,ARG002,F401,F841` passed during the unused-code cleanup, and Ruff/MyPy passed on touched files. The earlier full-suite validation baseline from the prior remediation pass was `uv run pytest` passing (`362 passed`). The earlier full-coverage baseline before this remediation wave was `uv run pytest --cov=helpers --cov-report=term-missing` passing with overall `helpers` coverage at `72%`; coverage should be re-measured again before the next coverage-first session.
- Recent scientific-validity remediation work completed a 10-phase hardening pass across Stage 4.1, 4.2, 5, 7, 8, 10, 11, and 12. When resuming scientific review or follow-up implementation, assume the following are now intentionally true unless the user explicitly requests otherwise:
  - Stage 12 hard predictions now use the declared recipe threshold instead of a hidden fixed threshold.
  - Stage 5 is non-destructive by default; destructive move behavior requires explicit opt-in.
  - Stage 2 patch filenames are deterministic and derived from stable identifiers instead of timestamps/random suffixes.
  - Stage 4.2 graph tuning uses patient/group-aware splitting for inner CV and held-out evaluation.
  - Stage 11 splits validation patients before recipe optimization and defaults to fairness-aware evaluation with `spatial_patient_policy="all"`.
  - Stage 11 now uses a 3-way patient-disjoint split inside `VALIDATION` for optimization, calibration, and optional development holdout.
  - Stage 11 now tunes `roi_gate_threshold` during optimization and calibrates a separate `decision_threshold` on the internal calibration subset using patient-level `MCC`.
  - Stage 12 now treats ROI gating and final hard-decision thresholding as separate semantics and fails closed on legacy one-threshold recipes.
  - Stage 5 now enforces downstream-aware validation sizing guardrails for the Stage 11 redesign, with a hard minimum target of `30` validation patients total and, when the dataset can support it, at least `15` positive and `15` negative validation patients.
  - Stage 7 packaging and Stage 8 smart sampling include stale-output protection, but cross-stage invisible reuse risks should still be reviewed carefully before trusting reused artifacts for scientific claims.
  - Stage 5 no longer allows `CROSSFOLD_OBJECTIVE_SCORE_SPLIT=TEST`.
  - Stage 4.1 optimization sampling is patient-aware rather than patch-naive.
  - Training and LR finder now default to `PAPER` execution mode instead of `FAST_DEV`.
  - Shared runtime provenance now flows through `helpers/provenance.py` into Stage 9, Stage 10 metadata, and Stage 12 run configs.
  - The forward migration plan is now: Stage 2 may stage one external-SSD WSI at a time into local SSD temp storage for processing; Stage 4.3 remains the last PNG-based stage; Stage 5 builds one cleaned source HDF5 dataset immediately after Stage 4.3; Stage 6 and Stage 7 are intended to be HDF5-native downstream of that boundary.
  - Under the HDF5-native Stage 6 design, stain normalization remains scientifically train-fitted only: entropy-based template selection and parameter fitting must use `TRAIN` rows only, and the resulting frozen normalizer must be applied to `TRAIN`, `VALIDATION`, and `TEST`.
- Recent cleanup follow-up completed the documentation-aligned HDF5 simplification for downstream data preparation: Stage 6 and Stage 7 no longer carry PNG-era backward-compatibility branches, split-stat recomputation is centralized in `helpers/crossfold/provenance.py`, and shared stage logger setup now lives in `helpers/logging_utils.py`.
- Important remaining scientific risks after the remediation pass, and the best starting points for the next session, are:
  - Highest remaining Stage 11 risk is no longer validation reuse by itself; Stage 11 is development-only. The highest remaining Stage 11 risk is unfair candidate mixing if provenance-compatible candidate gating is not enforced. Start in `helpers/ensemble_optimizer/metadata.py`, `helpers/ensemble_optimizer/pipeline.py`, and `helpers/training/checkpointing.py`.
  - Highest remaining Stage 12 risk is no longer threshold-semantic ambiguity; that redesign is now in place. The highest remaining Stage 12 risk is invisible stale-artifact reuse across Stage 7 packaging, Stage 8 smart sampling, Stage 11 recipe generation, and Stage 12 inference handoff when file paths remain stable but contents change. Strengthen content-based provenance checks in `helpers/packaging/writer.py`, `helpers/smart_sampling/writer.py`, `helpers/ensemble_optimizer/reporting.py`, and `helpers/ensemble_inference/pipeline.py`.
  - Operational follow-up after the threshold redesign: regenerate Stage 5 outputs under the new validation sizing guardrails, rerun Stage 11 to emit dual-threshold recipes, and rerun Stage 12 using only the new recipes.
  - Medium remaining risk: Stage 2 patch extraction still appears center-based before overlap filtering, which may bias the dataset toward lesion cores and under-sample boundary patches. If future work targets sampling bias in extraction, start in `helpers/extraction/patch_engine.py`.
- Important remaining Stage 2 performance risk after the latest optimization pass is no longer candidate filtering first. The best next optimization target is `read_region` / patch-output strategy inside `helpers/extraction/patch_engine.py`, because representative profiling showed `read_region` dominating full-slide runtime after the geometry optimizations landed.
- Recent targeted Stage 11/12 coverage work added focused tests in `tests/test_ensemble_inference_metrics.py`, `tests/test_ensemble_inference_data.py`, `tests/test_ensemble_inference_inference.py`, `tests/test_ensemble_optimizer_data.py`, and `tests/test_ensemble_optimizer_optimization.py`.
- The latest targeted coverage snapshot for those five priority helpers, using `uv run coverage run -m pytest ...` followed by `uv run coverage report -m ...`, is: `helpers/ensemble_inference/metrics.py` 100%, `helpers/ensemble_inference/data.py` 95%, `helpers/ensemble_inference/inference.py` 98%, `helpers/ensemble_optimizer/data.py` 95%, and `helpers/ensemble_optimizer/optimization.py` 79%, for 90% combined coverage across those target modules.
- Highest-priority remaining coverage work is no longer Stage 11/12-first. Start with the largest repo-wide coverage draggers: zero-coverage legacy-heavy modules and very low-coverage core helpers.
- When resuming coverage work, reuse the new lightweight patterns already added: tiny memmap caches, fake Optuna studies, CPU-only fake models, and direct monkeypatching of ROI generation and AUPRC helpers instead of real Stage 11 end-to-end runs.
- Current highest-priority coverage queue for the next session is:
  - zero-coverage modules first: `helpers/artifact/logging.py`, `helpers/artifact/model_loader.py`, `helpers/artifact/processor.py`, `helpers/optimization_sampling/logging.py`, `helpers/sanity/reporting.py`, and `helpers/wsi/*.py`
  - then very low-coverage high-impact modules: `helpers/extraction/patch_engine.py` (`20%`), `helpers/extraction/data_handlers.py` (`16%`), `helpers/lr_finder/runner.py` (`19%`), `helpers/ensemble_optimizer/models.py` (`20%`), and `helpers/smart_sampling/embeddings.py` (`37%`)
  - then medium-priority modules still below 80% such as `helpers/ensemble_inference/pipeline.py`, `helpers/smart_sampling/selection.py`, `helpers/sanity/manifest_checks.py`, `helpers/crossfold/splitting.py`, and `helpers/sanity/disk_checks.py`
- Latest repo-wide low-coverage snapshot from `uv run pytest --cov=helpers --cov-report=term-missing`:
  - `helpers/artifact/logging.py` 0%
  - `helpers/artifact/model_loader.py` 0%
  - `helpers/artifact/processor.py` 0%
  - `helpers/optimization_sampling/logging.py` 0%
  - `helpers/sanity/reporting.py` 0%
  - `helpers/wsi/colors.py` 0%
  - `helpers/wsi/maps.py` 0%
  - `helpers/wsi/process.py` 0%
  - `helpers/wsi/slide_info.py` 0%
  - `helpers/wsi/tis_detect_helper_fx.py` 0%
  - `helpers/extraction/data_handlers.py` 16%
  - `helpers/extraction/patch_engine.py` 20%
  - `helpers/lr_finder/runner.py` 19%
  - `helpers/ensemble_optimizer/models.py` 20%
  - `helpers/smart_sampling/embeddings.py` 37%

## Skills

- Repository-local skills are available under `skills/` and should be used when the task matches their scope.
- Use `skills/scientific-code-to-latex/SKILL.md` when converting repository behavior into publication-ready scientific prose or LaTeX methods text grounded in the code.
- Use `skills/scientific-validation-review/SKILL.md` when reviewing training, evaluation, splitting, preprocessing, or other research code for threats to scientific validity, leakage, bias, or irreproducibility.
- Use `skills/python-performance-optimization/SKILL.md` when profiling or optimizing Python performance, especially for scientific workloads, heavy loops, pandas/NumPy code, or parallel execution.
- For optimization work, prefer representative samples located under `PROJECTS_BASE_PATH` from `.env` for profiling and benchmarking whenever those project samples are available.

## Naming Conventions

- Match existing domain terminology: `cancer`, `not_cancer`, `patient`, `split`, `manifest`, `artifact`, `annotation`.
- Preserve output folder names exactly when they are part of the pipeline contract.
- Do not silently rename `TRAIN`, `VALIDATION`, `TEST`, `CANCER`, `NOT_CANCER`, `CANCER_MASK`, or `NOT_CANCER_MASK`.
- Reserve ALL_CAPS names for environment variables, true constants, and registry/config keys; prefer `snake_case` for regular runtime variables.

## Logging and Output

- Prefer `logging` for nontrivial workflows and long-running scripts.
- Many scripts configure root logging with both file and console handlers; follow that pattern when extending those files.
- Avoid excessive `print` in modern/refactored scripts unless the file already provides a terminal UX.
- Allow styled terminal output with ANSI sequences and emoji to make it user-friendly and colorful.
- For Stage 1 artifact detection, preserve the current colorful console + file logging pattern instead of reverting to plain output.

## Data Integrity Rules

- Preserve patient-level split isolation; avoid anything that can introduce leakage across `TRAIN`, `VALIDATION`, and `TEST`.
- Preserve image/mask pairing by filename.
- Preserve manifest schema and run metadata unless the user explicitly requests schema changes.
- Preserve label semantics: cancer remains positive class, not-cancer remains negative class.
- Do not change patch acceptance thresholds, stain normalization behavior, or artifact filtering semantics without clear intent.
- Do not change the approved Stage 8 architecture/encoder matrix, registry defaults, or training semantics without explicit user intent.

## Filesystem and Artifact Safety

- This repo produces many large/generated outputs: PNG patches, logs, manifests, entropy caches, and HDF5 files.
- Avoid committing generated data unless the user explicitly asks.
- Treat `.env`, `database.db`, `DOWNLOAD_DRIVE/credentials.json`, `credentials.json`, and `token.json` as sensitive or local-state files.
- Treat Aim repositories, training checkpoints, model weights, HDF5 files, manifests, and logs as generated artifacts unless the user explicitly asks to commit them.
- Respect `.gitignore`, which already ignores `.env`, logs, PNGs, bytecode, and credential files.
- Avoid deleting source folders or generated datasets unless the user explicitly requests destructive cleanup.

## Agent-Specific Guidance

- Prefer minimal, local edits over broad rewrites.
- First understand whether a file is legacy, helper, or actively used in the main pipeline.
- Do not introduce a new framework or project-wide tooling without an explicit request.
- If adding a new command to documentation, make sure it is actually supported by the repository as-is.
- When proposing validation, clearly distinguish between official repo workflows and ad hoc checks.
- If you touch scientific logic, be conservative and preserve reproducibility.
- If you touch split generation or sanity checks, assume correctness matters more than cleverness.
- If you touch Stage 8 training, preserve the helper-first structure and keep `10_training_ensemble.py` orchestration-focused.

## 1. Purpose

This file defines the architectural, engineering, testing, and implementation rules that all agents must follow when modifying this project.

The objective is to guarantee:

- clean separation of concerns
- deterministic behavior where appropriate
- high test coverage
- safe refactoring
- maintainable Python code
- minimal reinvention of existing solutions

This document is normative. Agents must treat these rules as requirements, not suggestions.

---

## 2. Core Engineering Principles

This project follows:

- **Clean Architecture**
- **Separation of Concerns (SoC)**
- **Single Responsibility Principle (SRP)**
- **Extreme Programming (XP)**
- **Test-Driven Development (TDD)**
- **Fail Fast**
- **Prefer composition over unnecessary complexity**

Agents must preserve these properties in every implementation.

---

## 3. Technology Stack

The project stack uses:

- **Python**
- **uv** for dependency and environment management
- **pytest** for tests
- **pytest-cov** for coverage
- **ruff** for linting
- **mypy** for static typing
- **python-dotenv** for environment configuration
- **argparse** for parameter input

Agents must use the project’s existing stack unless there is a strong technical reason to introduce a new dependency.

Libraries and behaviors must be defined in `pyproject.toml`.

---

## 4. Dependency Management Rules

All dependencies must be managed with **uv**.

### Required commands

Install runtime dependency:

uv add <package>

Install development dependency:

uv add --dev <package>

Run Python commands inside the project environment:

uv run <command>

Examples:


`uv run pytest`
`uv run pytest --cov=helpers --cov-report=term-missing`
`uv run ruff check .`
`uv run mypy .`

Dependency policy

Agents must:

- add dependencies only when necessary

- prefer mature, widely used libraries (especially NumPy, pandas, or Polars over coding from scratch)

- avoid adding dependencies for trivial functionality already available in the standard library or existing project stack

- keep the dependency graph as small as possible

Agents must not reinvent the wheel when a stable Python library already solves the problem well.

---

## 5. Project Structure

MASTER_CODING:
- `1_artifact_detection.py`
- `2_database_manager.py`
- `3_1_imageReader.py`
- `4_1_optimization_sampling.py`
- `4_2_tune_graph_method.py`
- `4_3_cleaner_script.py`
- `5_pack_splits_to_hdf5.py`
- `6_crossfold.py`
- `7_sanity_checks.py`
- `8_smart_sampler.py`
- `9_lr_finder.py`
- `10_training_ensemble.py`
- `11_optimizer_ensemble.py`
- `12_inference_ensemble.py`
- `.env`
- `.env_example`
- `.gitignore`
- `AGENTS.md`
- `README.md`
- `pyproject.toml`
- `setup_colab.sh`
- `setup_windows.ps1`
- `colab_setup.md`
- `Dockerfile`
- `.devcontainer/devcontainer.json`
- `training_model_registry.json`
- `skills/`
- `helpers/__init__.py`
- `helpers/artifact/__init__.py`
- `helpers/artifact/config.py`
- `helpers/artifact/logging.py`
- `helpers/artifact/model_loader.py`
- `helpers/artifact/paths.py`
- `helpers/artifact/pipeline.py`
- `helpers/artifact/processor.py`
- `helpers/artifact/repository.py`
- `helpers/artifact/zip.py`
- `helpers/extraction/__init__.py`
- `helpers/extraction/artifact_index.py`
- `helpers/extraction/config.py`
- `helpers/extraction/repository.py`
- `helpers/extraction/image_reader_service.py`
- `helpers/extraction/data_handlers.py`
- `helpers/extraction/patch_engine.py`
- `helpers/graph/__init__.py`
- `helpers/graph/cleaning_config.py`
- `helpers/graph/cleaning_pipeline.py`
- `helpers/graph/contamination.py`
- `helpers/graph/parameter_store.py`
- `helpers/graph/tuning_config.py`
- `helpers/graph/tuning_pipeline.py`
- `helpers/crossfold/__init__.py`
- `helpers/crossfold/config.py`
- `helpers/crossfold/discovery.py`
- `helpers/crossfold/entropy.py`
- `helpers/crossfold/io.py`
- `helpers/crossfold/logging.py`
- `helpers/crossfold/normalization.py`
- `helpers/crossfold/pipeline.py`
- `helpers/crossfold/provenance.py`
- `helpers/crossfold/splitting.py`
- `helpers/optimization_sampling/__init__.py`
- `helpers/optimization_sampling/config.py`
- `helpers/optimization_sampling/logging.py`
- `helpers/optimization_sampling/overlay.py`
- `helpers/optimization_sampling/pipeline.py`
- `helpers/optimization_sampling/sampling.py`
- `databases/`
- `databases/database.db`
- `helpers/runtime_platform.py`
- `helpers/training/__init__.py`
- `helpers/training/checkpointing.py`
- `helpers/training/config.py`
- `helpers/training/data.py`
- `helpers/training/gpu.py`
- `helpers/training/loop.py`
- `helpers/training/losses.py`
- `helpers/training/metrics.py`
- `helpers/training/models.py`
- `helpers/training/pipeline.py`
- `helpers/training/registry.py`
- `helpers/training/reporting.py`
- `helpers/training/runtime.py`
- `helpers/training/utils.py`
- `helpers/wsi/__init__.py`
- `helpers/wsi/colors.py`
- `helpers/wsi/maps.py`
- `helpers/wsi/process.py`
- `helpers/wsi/slide_info.py`
- `helpers/wsi/tis_detect_helper_fx.py`
- `logs/`
- `tests/`

---

## 6. Architectural Boundaries

Each root script listed below is a monolithic, self-contained script:

- 1_artifact_detection.py
- 2_database_manager.py
- 3_1_imageReader.py
- 4_1_optimization_sampling.py
- 4_2_tune_graph_method.py
- 4_3_cleaner_script.py
- 5_pack_splits_to_hdf5.py
- 6_crossfold.py
- 7_sanity_checks.py
- 8_smart_sampler.py
- 9_lr_finder.py
- 10_training_ensemble.py
- 11_optimizer_ensemble.py
- 12_inference_ensemble.py

These scripts perform actions independently of each other, except for `2_database_manager.py` and `3_1_imageReader.py`, which interact with each other and with `helpers/extraction/data_handlers.py` and `helpers/extraction/patch_engine.py`.

---

## 7. Mandatory Testing Policy

- Testing is mandatory for all new behavior.

**Absolute rule**

Every new function, behavior change, bug fix, or edge-case handling must include tests.

**TDD workflow**

Agents must follow this sequence:

- write a failing test

- run the tests and verify the test fails for the expected reason

- implement the minimal code necessary

- rerun the tests and verify they pass

- refactor only after the tests are green

**Validity rule**

- If the test did not fail before implementation, the validation is incomplete.

Minimum expectation

- every public function must have direct tests

- critical private logic should be covered through behavior tests

- bug fixes must include a regression test

- edge cases must be tested explicitly

## 9. Coverage Rules

- Coverage is a quality gate.

**Coverage target**

- minimum target: 90% line coverage in the core project

- all new code should be covered

- untested branches in critical logic are not acceptable without justification

Required command

`uv run pytest --cov=helpers --cov-report=term-missing`

Agents should use the missing-lines report to identify coverage gaps before finishing.

---

## 10. Test Design Rules

Tests must be:

- deterministic

- isolated

- readable

- small in scope when unit tests

- explicit about expected behavior

**Tests must avoid**

- network calls

- external APIs

- flaky timing assumptions

- hidden state dependencies

- dependence on unrelated files unless explicitly integration-tested

**Preferred patterns**

- simple fixture factories

- parametrized tests for rule matrices

- mocks only at true infrastructure boundaries

- direct assertion of observable behavior


```python
def test_classify_usage_rejects_when_usage_exceeds_threshold() -> None:
    result = classify_usage(cpu_used=80.0, cpu_requested=50.0)
    assert result == "Rejected"
```


---

## 11. Python Coding Standards

All code must follow modern Python best practices.

**Required standards**

- PEP 8 compliance

- meaningful naming

- explicit imports

- type hints on all public functions

- docstrings on public functions

- small, composable functions

- low cyclomatic complexity where practical

**Forbidden patterns**

- wildcard imports

- dead code

- commented-out production code

- misleading names

- giant functions with mixed responsibilities

- broad except Exception without justification

- silent failure paths

**Function design guidance**

Prefer functions that:

- do one thing well

- accept explicit inputs

- return explicit outputs

- are easy to test in isolation

---

## 12. Typing Policy

- Type hints are required.

**Agents must**:

- annotate function parameters

- annotate return types

- prefer precise standard types

- use TypedDict, Protocol, dataclass, or clear domain objects where they improve clarity

Run static checks with:

uv run mypy .

If a library lacks typing support, prefer installing type stubs when appropriate rather than weakening the entire typing discipline.

---

## 13. Linting and Formatting Policy

- Code quality checks are mandatory.

**Required commands**

`uv run ruff check .`
`uv run ruff format .`


Lint issues must be resolved before finalizing changes.

## 14. Error Handling Policy

- The code must fail early and fail clearly.

**Rules**

- validate inputs as early as practical

- raise meaningful exceptions

- avoid swallowing errors

- include contextual information in error messages

- distinguish domain errors from infrastructure errors when useful

---

## 15. Logging and Observability

- Implementations should be debuggable.
- All logs should be saved in `/logs`.

**Agents should prefer**:

- structured log messages

- explicit error context

- stable identifiers in logs where useful

- observability at important boundaries


## 16. Refactoring Rules

- Refactoring is encouraged, but must be disciplined.

**Agents may refactor to improve**:

- readability

- testability

- modularity

- duplication

- naming

- architecture alignment

**Agents must not refactor in a way that**:

- changes behavior unintentionally

- removes tests

- mixes unrelated concerns

- introduces speculative abstractions with no present need

**Safe refactoring sequence**

- ensure tests are green

- refactor incrementally

- rerun tests

- keep behavior unchanged unless the task explicitly requires a behavior change

---

## 17. Implementation Constraints for Agents

When implementing or changing code, agents must obey the following operational rules.

### 17.1 Before writing code

Agents must first:

- understand which layer owns the behavior

- determine whether a library already solves the problem

- identify the tests that need to be added or updated

### 17.2 During implementation

Agents must:

- keep changes minimal and focused

- preserve architecture boundaries

- avoid mixing concerns across modules

- add or update tests alongside the code

### 17.3 After implementation

Agents must verify:

- tests pass

- lint passes

- typing passes

- coverage remains acceptable

- no unrelated code was changed unnecessarily

---

## 18. Anti-Patterns

The following are prohibited unless explicitly justified.

**Architecture anti-patterns**

- formatting logic spread across unrelated modules

**Testing anti-patterns**

- implementing code without tests

- tests that assert implementation details instead of behavior

- flaky tests

- tests that depend on execution order

**Code anti-patterns**

- giant procedural functions

- copy-paste duplication

- mutable global state

- magic constants scattered through the code

- deep nesting when guard clauses would be clearer

---

## 19. Preferred Design Heuristics

Agents should prefer:

- pure functions for transformation logic

- dataclass for clear structured data where appropriate

- constants for thresholds and repeated literals

- small adapters around infrastructure dependencies

- thin orchestration and rich helper modules

- explicit contracts between pipeline stages

- use abstraction only when it simplifies the system. Do not introduce unnecessary indirection.

---

## 20. Example Delivery Checklist

Before considering a task complete, agents must ensure the answer to each item is yes.

- Was the correct module chosen for the change?

- Was a failing test written first?

- Does every new behavior have tests?

- Do all tests pass?

- Does lint pass?

- Does typing pass?

- Is the implementation PEP-compliant?

- Was an existing library used instead of custom reinvention when appropriate?

- Were architecture boundaries preserved?

- Is the code easier to understand than before?

- If any answer is no, the task is incomplete.

---

## 21. Recommended Developer Commands

`uv sync`
`uv run pytest`
`uv run pytest --cov=helpers --cov-report=term-missing`
`uv run ruff check .`
`uv run ruff format .`
`uv run mypy .`


If only a subset is needed during local iteration, agents may scope commands appropriately, but final validation should cover the full affected surface.


## 22. Definition of Done

A task is complete only when all of the following are true:

- the requested behavior is implemented

- tests covering the behavior exist

- the tests fail before implementation and pass after implementation

- linting passes

- typing checks pass

- architecture rules are respected

- no unnecessary complexity was introduced

- the solution uses established Python best practices

If these conditions are not met, the implementation is not complete.

---

## 24. Final Rule

Agents must optimize for:

correctness first

clarity second

maintainability third

performance where relevant and measurable

Do not trade correctness and testability for cleverness.
