# AGENTS.md
Agent guide for this repository. Keep changes small, scientific, and test-backed.

## Scope
- Repository type: Python research pipeline for pathology whole-slide-image processing.
- Architecture: script-first root entrypoints with domain helpers under `helpers/<domain>/`.
- Main stages: artifact detection, extraction, cleaning, HDF5 packaging, crossfolding, sanity checks, sampling, training, ensemble optimization, inference.
- Priority order: correctness > reproducibility > maintainability > performance.

## Rules Sources
- Existing `/workspace/AGENTS.md` was replaced with this shorter agent-focused guide.
- No `.cursorrules` file was found.
- No `.cursor/rules/` directory was found.
- No `.github/copilot-instructions.md` file was found.

## Environment
- Python version: `3.12`.
- Dependency manager: `uv`.
- Linter/formatter: `ruff`.
- Type checker: `mypy` with `strict = true` and `ignore_missing_imports = true`.
- Test runner: `pytest` with tests in `tests/`.
- Ruff line length: `100`.

## High-Value Commands

### Setup
- Sync runtime deps: `uv sync --python 3.12`
- Sync with dev deps: `uv sync --python 3.12 --group dev`

### Tests
- Run full suite: `uv run pytest`
- Run coverage for helpers: `uv run pytest --cov=helpers --cov-report=term-missing`
- Run one test file: `uv run pytest tests/test_some_module.py`
- Run one test by node id: `uv run pytest tests/test_some_module.py::test_specific_case`
- Run one test class: `uv run pytest tests/test_some_module.py::TestSomething`
- Filter tests by name: `uv run pytest -k "artifact and not slow"`
- Stop on first failure: `uv run pytest -x`
- Show locals on failure: `uv run pytest -x -vv --showlocals`

### Lint / Format / Types
- Lint full repo: `uv run ruff check .`
- Lint touched files: `uv run ruff check path/to/file.py tests/test_file.py`
- Format full repo: `uv run ruff format .`
- Format touched files: `uv run ruff format path/to/file.py tests/test_file.py`
- Type-check full repo: `uv run mypy .`
- Type-check touched scope: `uv run mypy path/to/file.py tests/test_file.py`

### Running Entrypoints
- Run stage scripts through `uv`, for example: `uv run --python 3.12 python 6_crossfold.py`
- Most runtime configuration comes from `.env`; update `.env_example` when adding variables.

## Repository Shape
- Root scripts are standalone apps: `1_artifact_detection.py` through `12_inference_ensemble.py`.
- Domain logic belongs in `helpers/<domain>/`; keep root scripts orchestration-focused.
- Shared cross-domain utilities may live in top-level `helpers/`, e.g. `helpers/runtime_platform.py` and `helpers/logging_utils.py`.
- Tests belong in `tests/`; logs in `logs/`; databases in `databases/`.

## Working Norms
- First inspect the relevant stage script and its helper package.
- Reuse existing helpers before adding new modules.
- Preserve pipeline contracts unless the task explicitly changes them.
- Every behavior change needs tests; for bug fixes, add a regression test.
- Before finishing, run targeted tests for touched code at minimum.
- If shared infrastructure or scientific logic changes, run broader relevant suites too.

## Code Style

### Imports
- Use explicit imports; never use wildcard imports.
- Prefer standard library, then third-party, then local imports.
- Use package-safe imports from `helpers...`; do not use sibling imports like `from data_handlers import ...`.
- Remove unused imports.

### Formatting
- Follow Ruff formatting and a max line length of `100`.
- Prefer small functions, guard clauses, and shallow nesting.
- Avoid commented-out code and dead code.
- Use ASCII by default unless the file already requires Unicode.

### Types
- Add type hints to all public functions.
- Match repository style: `Path`, `Mapping`, `Sequence`, `tuple[...]`, `str | None`.
- Prefer precise standard types over `Any`.
- Use `dataclass(frozen=True)` for validated configuration objects when appropriate.
- Keep touched code compatible with strict mypy.

### Naming
- Use `snake_case` for functions, variables, and module-level helpers.
- Use `PascalCase` for classes.
- Reserve `ALL_CAPS` for true constants and environment variable names.
- Match repo vocabulary: `cancer`, `not_cancer`, `patient`, `split`, `manifest`, `artifact`, `annotation`.
- Do not silently rename contract names such as `TRAIN`, `VALIDATION`, `TEST`, `IMAGES`, `MASKS`, `REJECTED_IMAGES`, `REJECTED_MASKS`.

### Docstrings and Comments
- Add concise docstrings to public functions.
- Keep comments only for non-obvious scientific or control-flow reasoning.
- Do not add noisy comments that restate the code.

## Error Handling
- Fail fast on invalid configuration or broken contracts.
- Raise specific, meaningful exceptions with context.
- Validate environment variables early.
- Avoid broad `except Exception` unless you re-raise with useful context or are at a true process boundary.
- Do not silently swallow scientific integrity errors.

## Logging
- Prefer `logging` over `print` for nontrivial workflows.
- Reuse `helpers/logging_utils.py` for shared logger setup.
- Preserve existing user-facing logging patterns where already established, especially Stage 1.
- Write logs under `logs/`.

## Scientific and Data Integrity Rules
- Preserve patient-level split isolation.
- Preserve image/mask row alignment and filename parity.
- Preserve label semantics: cancer is positive (`1`), not-cancer is negative (`0`).
- Do not change stain normalization behavior, sampling semantics, or artifact logic without explicit intent.
- Preserve filename-keyed provenance joins, especially artifact metadata.
- In HDF5 workflows, keep canonical datasets stable: `images`, `masks`, `labels`, `patient_ids`, `filenames`.

## Stage Boundaries
- Keep Stage 1 logic in `helpers/artifact/*` and orchestration in `1_artifact_detection.py`.
- Keep Stage 2 orchestration in `2_database_manager.py` and `3_1_imageReader.py`; extraction details belong in `helpers/extraction/*`.
- Keep Stage 4.1 sampling logic in `helpers/optimization_sampling/*`.
- Keep Stage 4.2/4.3 graph contamination logic shared in `helpers/graph/contamination.py`.
- Keep Stage 5 packaging logic in `helpers/packaging/*`.
- Keep Stage 6 split and normalization logic in `helpers/crossfold/*`.
- Keep Stage 7 integrity checks in `helpers/sanity/*`.
- Keep Stage 8 smart-sampling logic in `helpers/smart_sampling/*`.
- Keep Stage 9-12 training and inference logic in their matching helper packages.

## Filesystem and Safety
- Do not commit generated artifacts unless explicitly asked.
- Treat `.env`, `credentials.json`, `token.json`, databases, checkpoints, HDF5 outputs, manifests, Aim repos, and logs as sensitive or generated.
- Respect `.gitignore`.
- Do not delete datasets, logs, or outputs unless the user explicitly asks.
- Never use destructive git commands like `git reset --hard` or `git checkout --` unless explicitly requested.

## Platform Guidance
- Reuse `helpers/runtime_platform.py` for OS/path handling.
- On Windows, `OPENSLIDE_PATH` must point to the OpenSlide `bin` folder.
- Avoid open-coded platform checks when shared helpers already exist.

## Validation Checklist
- Relevant tests added or updated.
- Relevant tests pass.
- `uv run ruff check` passes for touched files.
- `uv run ruff format` applied where needed.
- `uv run mypy` passes for touched files.
- No unrelated files changed.
- Pipeline contracts and scientific invariants preserved.

## Agent Heuristics
- Prefer minimal local edits over broad rewrites.
- Use existing libraries and helpers instead of reinventing behavior.
- Optimize only with evidence; for Stage 2 performance work, benchmark real samples before and after.
- If changing scientific logic, be conservative and prioritize reproducibility.
- If unsure where code belongs, prefer a thin orchestrator and a richer helper module.
