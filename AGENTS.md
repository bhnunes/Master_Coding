# AGENTS.md
Agent guide for coding agents working in this repository.

## Purpose
- Repository type: Python research pipeline for pathology whole-slide-image processing.
- Architecture: root stage scripts orchestrate work; domain logic lives under `helpers/<domain>/`.
- Primary goal: preserve scientific correctness and reproducibility while making minimal, well-tested changes.
- Priority order: correctness > reproducibility > maintainability > performance.

## Rule Sources
- This repository contains `/workspace/AGENTS.md`.
- No `.cursorrules` file was found.
- No `.cursor/rules/` directory was found.
- No `.github/copilot-instructions.md` file was found.

## Environment
- Python: `3.12`
- Dependency manager: `uv`
- Test runner: `pytest`
- Linter and formatter: `ruff`
- Type checker: `mypy`
- Mypy mode: `strict = true`, `ignore_missing_imports = true`
- Ruff line length: `100`
- Tests live under `tests/`

## Setup Commands
- Install runtime dependencies: `uv sync --python 3.12`
- Install runtime and dev dependencies: `uv sync --python 3.12 --group dev`

## Test Commands
- Run full test suite: `uv run pytest`
- Run coverage for helper modules: `uv run pytest --cov=helpers --cov-report=term-missing`
- Run one test file: `uv run pytest tests/test_artifact_pipeline.py`
- Run one test by node id: `uv run pytest tests/test_artifact_pipeline.py::test_pipeline_processes_all_pending_records_and_continues_after_failure`
- Run one test class: `uv run pytest tests/test_training_config.py::TestTrainingConfig`
- Run tests matching an expression: `uv run pytest -k "artifact and not slow"`
- Stop on first failure: `uv run pytest -x`
- Show verbose failures with locals: `uv run pytest -x -vv --showlocals`

## Lint, Format, and Type Commands
- Lint entire repo: `uv run ruff check .`
- Lint touched files only: `uv run ruff check path/to/file.py tests/test_file.py`
- Format entire repo: `uv run ruff format .`
- Format touched files only: `uv run ruff format path/to/file.py tests/test_file.py`
- Type-check entire repo: `uv run mypy .`
- Type-check touched scope: `uv run mypy path/to/file.py tests/test_file.py`

## Stage Entrypoints
- Stage 1: `1_artifact_detection.py`
- Stage 2: `2_database_manager.py`
- Stage 3: `3_pack_splits_to_hdf5.py`
- Stage 4.1: `4_1_optimization_sampling.py`
- Stage 4.2: `4_2_tune_graph_method.py`
- Stage 4.3: `4_3_cleaner_script.py`
- Stage 5: `5_crossfold.py`
- Stage 6: `6_sanity_checks.py`
- Stage 7: `7_smart_sampler.py`
- Stage 8: `8_lr_finder.py`
- Stage 9: `9_training_ensemble.py`
- Stage 10: `10_optimizer_ensemble.py`
- Stage 11: `11_inference_ensemble.py`
- Run a stage script with `uv`, for example: `uv run --python 3.12 python 5_crossfold.py`

## Repository Shape
- Keep root scripts orchestration-focused.
- Put reusable domain logic in `helpers/<domain>/`.
- Reuse shared helpers such as `helpers/runtime_platform.py` and `helpers/logging_utils.py`.
- Keep tests in `tests/`.
- Treat `logs/`, `databases/`, generated HDF5 outputs, manifests, checkpoints, and Aim repos as generated artifacts unless the task says otherwise.

## Working Norms
- Inspect the relevant stage script and helper package before changing behavior.
- Prefer the smallest correct change over broad rewrites.
- Reuse existing helpers before creating new modules.
- Preserve existing pipeline contracts unless the task explicitly changes them.
- For Stage 2 `.svs/.xml` work, verify whether `TAG=HISEG` is active before changing color or XML parsing behavior.
- Every behavior change needs tests.
- For bug fixes, add or update a regression test.
- Run targeted tests for touched code before finishing.
- If you change shared infrastructure or scientific logic, run a broader relevant slice of the suite.

## Imports
- Prefer `from __future__ import annotations` in Python modules, matching existing code.
- Use explicit imports only; never use wildcard imports.
- Order imports as standard library, third-party, then local packages.
- Use package-safe imports from `helpers...`; do not use sibling-relative shortcuts.
- Remove unused imports.

## Formatting
- Follow Ruff formatting and the repository max line length of `100`.
- Prefer small functions, guard clauses, and shallow nesting.
- Avoid commented-out code and dead code.
- Use ASCII unless the file already requires non-ASCII text.
- Keep orchestrators thin and push nontrivial logic into helper modules when it improves reuse or clarity.

## Types
- Add type hints to public functions and nontrivial helpers.
- Match repository conventions such as `Path`, `Mapping`, `Sequence`, `tuple[...]`, and `str | None`.
- Prefer precise types over `Any`.
- Use `@dataclass(frozen=True)` for validated configuration objects and immutable result containers when appropriate.
- Keep touched code compatible with strict mypy.

## Naming
- Use `snake_case` for functions, variables, and module-level helpers.
- Use `PascalCase` for classes.
- Use `ALL_CAPS` for constants and environment variable names.
- Match repository vocabulary: `cancer`, `not_cancer`, `patient`, `split`, `manifest`, `artifact`, `annotation`.
- Do not silently rename established contract names such as `TRAIN`, `VALIDATION`, `TEST`, `IMAGES`, `MASKS`, `REJECTED_IMAGES`, and `REJECTED_MASKS`.

## Docstrings and Comments
- Add concise docstrings to public functions and public dataclasses.
- Keep comments for non-obvious scientific assumptions, invariants, or control flow.
- Do not add noisy comments that only restate the code.

## Error Handling
- Fail fast on invalid configuration, bad inputs, or broken pipeline contracts.
- Raise specific exceptions with useful context.
- Validate environment variables early.
- Prefer explicit validation over silent fallback behavior for scientific or data-integrity concerns.
- Avoid broad `except Exception` unless you are at a true process boundary or you re-raise with actionable context.
- Do not silently swallow scientific integrity errors.

## Logging
- Prefer `logging` over `print` for nontrivial workflows.
- Reuse `helpers/logging_utils.py` for logger setup and log-folder resolution.
- Preserve user-facing logging patterns where an entrypoint already depends on them.
- Write logs under `logs/` unless an existing stage contract dictates another path.

## Configuration
- Most runtime configuration comes from environment variables and `.env`.
- Update `.env_example` when adding or renaming environment variables.
- Use `helpers.runtime_platform.resolve_env_path` and related helpers for path-like environment variables.
- Preserve cross-platform behavior, especially Windows `OPENSLIDE_PATH` handling.
- `TAG=HISEG` enables the HISEG-specific Stage 2 SVS/XML annotation path.
- `TAG=Chile` enables the CHILE-specific Stage 2 SVS/XML annotation path.
- Supported `.svs/.xml` datasets resolve label colors internally in code; unsupported tags should fail explicitly.
- Stage 2 performance-related env vars now include:
  - `STAGE2_HDF5_COMPRESSION` with allowed values `gzip`, `lzf`, `none`
  - `OPENSLIDE_CACHE_BYTES` with default `0`
  - `STAGE2_PRELOAD_SCAN_AREA_MAX_BYTES` with default `0`
- Current best-known Stage 2 runtime choice in `.env` is `STAGE2_HDF5_COMPRESSION=none`.

## Scientific and Data Integrity Rules
- Preserve patient-level split isolation.
- Preserve image and mask row alignment.
- Preserve filename parity and filename-keyed provenance joins.
- Preserve label semantics: cancer is positive (`1`), not-cancer is negative (`0`).
- Do not change stain normalization, sampling semantics, artifact logic, or contamination logic without explicit intent.
- For HISEG XML annotations, preserve the hardcoded label policy: cancer colors map to positive, not-cancer colors map to negative, and rejected colors are skipped entirely.
- In HDF5 workflows, keep canonical dataset names stable: `images`, `masks`, `labels`, `patient_ids`, `filenames`.

## Stage Boundaries
- Keep Stage 1 logic in `helpers/artifact/*`.
- Keep extraction and image-reading details in `helpers/extraction/*`.
- Keep Stage 2 dataset-specific XML parsing localized to `helpers/extraction/data_handlers.py` and the Stage 2 request flow.
- Keep Stage 3 packaging logic in `helpers/packaging/*`.
- Keep Stage 4.1 sampling logic in `helpers/optimization_sampling/*`.
- Keep Stage 4.2 and 4.3 graph contamination logic shared in `helpers/graph/contamination.py` and related graph helpers.
- Keep Stage 5 split and normalization logic in `helpers/crossfold/*`.
- Keep Stage 6 integrity checks in `helpers/sanity/*`.
- Keep Stage 7 smart-sampling logic in `helpers/smart_sampling/*`.
- Keep Stage 8 learning-rate finder logic in `helpers/lr_finder/*`.
- Keep Stage 9 training logic in `helpers/training/*`.
- Keep Stage 10 ensemble optimization logic in `helpers/ensemble_optimizer/*`.
- Keep Stage 11 inference logic in `helpers/ensemble_inference/*`.

## Filesystem and Safety
- Respect `.gitignore`.
- Do not commit generated artifacts unless explicitly asked.
- Treat `.env`, credentials files, databases, HDF5 outputs, manifests, logs, checkpoints, and Aim repos as sensitive or generated.
- Do not delete datasets, logs, or outputs unless explicitly requested.
- Never use destructive git commands such as `git reset --hard` or `git checkout --` unless explicitly requested.

## Validation Checklist
- Relevant tests were added or updated.
- Relevant targeted tests pass.
- `uv run ruff check` passes for touched files.
- `uv run ruff format` has been applied where needed.
- `uv run mypy` passes for touched files.
- No unrelated files were modified intentionally.
- Pipeline contracts and scientific invariants remain intact.

## Stage 2 Performance Notes
- Use `stage2_performance_findings.md` as the canonical handoff document for Stage 2 benchmarking and optimization work.
- Use `uv run python scripts/benchmark_stage2_cases.py --case-ids 8,4,6,7,2,5,1` for the agreed benchmark set.
- Benchmark outputs live under `analysis/stage2_benchmarks*/`.
- The biggest confirmed Stage 2 wins so far were:
  - batched HDF5 shard writes in `helpers/extraction/hdf5_storage.py`
  - `STAGE2_HDF5_COMPRESSION=none`
  - replacing hot PIL-to-NumPy `np.array(...)` conversions with `np.asarray(...)` in `helpers/extraction/patch_engine.py`
- Current remaining extraction hotspot is `helpers/extraction/patch_engine.py`, especially `read_region()` inside `parallel_processing`.
- Negative results already established:
  - `OPENSLIDE_CACHE_BYTES=134217728` and `536870912` did not beat `0` after rebuilding with native OpenSlide `4.0.0`
  - scan-area preload was much slower
  - a streaming `source_signature` rewrite was slower and was reverted
- Docker/OpenSlide status:
  - The previous Ubuntu 22.04 package install exposed native OpenSlide `3.4.1`
  - `Dockerfile` now builds native OpenSlide `4.0.0` from source and includes a build-time self-check that fails unless native OpenSlide is `>= 4.0.0` and `OpenSlideCache` can be instantiated
  - The rebuilt container now reports native OpenSlide `4.0.0`, and `openslide.OpenSlideCache(...)` instantiates successfully
  - Despite that, `OPENSLIDE_CACHE_BYTES=0` remains the best-known setting on the agreed benchmark slide set

## Agent Heuristics
- Prefer minimal local edits.
- Prefer existing helpers and established patterns over reinvention.
- If changing scientific logic, be conservative and explicit.
- If performance work is requested, benchmark before and after when feasible.
- If unsure where code belongs, prefer a thin stage script and a richer helper module.
