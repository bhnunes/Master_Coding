# AGENTS.md

This file gives coding agents repository-specific guidance for working safely and effectively in this codebase.

## Purpose

- This repository is a Python research pipeline for pathology whole-slide-image processing.
- Main responsibilities include patch extraction, dataset splitting, scientific sanity checks, and HDF5 packaging.
- The repo is script-driven rather than package-driven.
- There is no formal build system, test suite, or linter configuration checked into the repo.

## Repository Layout

- `2_database_manager.py`: top-level orchestration for ingestion, SQLite tracking, and per-case processing.
- `3_1_imageReader_refactored.py`: per-slide worker CLI; selects the correct annotation handler and runs extraction.
- `data_handlers.py`: annotation format adapters for `.svs/.xml`, `.ndpi/.ndpa`, and JSON-based formats.
- `patch_engine.py`: main patch extraction engine; tissue checks, polygon masking, artifact filtering, image/mask writes.
- `5_crossfold_v6.py`: patient-level dataset split creation plus optional stain normalization.
- `6_sanity_checks_v3.py`: scientific integrity and dataset consistency checks.
- `7_pack_splits_to_hdf5.py`: converts prepared split folders into `TRAIN.h5`, `VALIDATION.h5`, and `TEST.h5`.
- `HELPERS/`: one-off utilities, legacy scripts, conversions, cleanup, and ad hoc support code.
- `background_experiment/`: separate experiment workflow with its own requirements.

## Rules Files

- No repository-local Cursor rules were found in `.cursor/rules/`.
- No `.cursorrules` file was found.
- No Copilot instructions file was found at `.github/copilot-instructions.md`.
- If such files are added later, follow them in addition to this document.

## Environment Setup

- Use Python 3.11+ if possible; this repo already contains `__pycache__` artifacts for Python 3.11.
- Install main dependencies with:

```bash
python -m pip install -r requirements.txt
```

- Install background experiment dependencies separately when needed:

```bash
python -m pip install -r background_experiment/requirements.txt
```

- Several scripts require a populated `.env` file.
- `.vscode/settings.json` enables `python.terminal.useEnvFile`, so local terminals may auto-load `.env`.
- Important env vars used by the main pipeline include:
  - `OPENSLIDE_PATH`
  - `TAG`
  - `SQLITE_DB_PATH`
  - `PROJECTS_BASE_PATH`
  - `IMAGE_READER_PATH`
  - `PYTHON_PATH`
  - `WINDOW_SIZE`
  - `STRIDE`
  - `MATCH_PERCENTAGE`
  - `TISSUE_PERCENTAGE`
  - `USE_ADVANCED_ARTIFACT_FILTERING`
  - `ACTIVATE_SANITY_CHECK_GEOJSON`
  - `GEOJSON_PATH`

## Build / Run Commands

- There is no package build step, `Makefile`, `tox`, or `pyproject.toml` workflow in this repo.
- The real execution model is running standalone Python scripts.
- Common commands:

```bash
python 2_database_manager.py
python 5_crossfold_v6.py
python 6_sanity_checks_v3.py
python 7_pack_splits_to_hdf5.py
```

- The most focused executable entrypoint is the single-slide extractor:

```bash
python 3_1_imageReader_refactored.py \
  --path_Image <slide_path> \
  --annotation_path <annotation_path> \
  --path_cancer_folder <out_cancer_dir> \
  --path_not_cancer_folder <out_not_cancer_dir> \
  --path_cancer_mask_folder <out_cancer_mask_dir> \
  --path_not_cancer_mask_folder <out_not_cancer_mask_dir> \
  --cancer_color <line_color_or_label> \
  --not_cancer_color <line_color_or_label> \
  --patient <patient_id>
```

- Optional artifact input for the worker script:

```bash
python 3_1_imageReader_refactored.py ... --path_artifacts_geojson <geojson_path>
```

## Lint / Format / Type Check Status

- No repo-configured `black`, `ruff`, `flake8`, `pylint`, `isort`, or `mypy` setup was found.
- Do not invent formatting or lint rules and apply them repo-wide unless asked.
- Keep edits narrow and aligned with the style already used in the file you are touching.
- Safe ad hoc syntax checks agents may run:

```bash
python -m py_compile 2_database_manager.py 3_1_imageReader_refactored.py patch_engine.py data_handlers.py
python -m compileall .
```

- Treat these as optional verification helpers, not as an official test suite.

## Test Commands

- There is no in-repo unit test framework.
- No `pytest`, `unittest`, or dedicated test directories were found.
- For this repo, “testing” usually means running the relevant script against a real sample or prepared dataset.

### Closest Equivalent to Running a Single Test

- Best single-target verification: run `3_1_imageReader_refactored.py` on one image/annotation pair.
- Best dataset-level verification: run `6_sanity_checks_v3.py` on one prepared dataset directory.
- Note that `6_sanity_checks_v3.py` is configured in its `if __name__ == "__main__":` block rather than a true CLI.
- `5_crossfold_v6.py` and `7_pack_splits_to_hdf5.py` are also configured primarily by editing the `__main__` configuration block.

## Configuration Conventions

- Preserve the existing configuration style of the file you edit.
- Operational scripts often read from `.env`.
- Analysis and packaging scripts often use hardcoded config constants inside `__main__`.
- Do not refactor a script from config-by-edit to argparse unless the user asks.
- Normalize filesystem paths where the script already does so.
- Be careful with Windows-specific paths such as `D:\...` and `C:\...`.

## Import Conventions

- Prefer `stdlib` imports first, then third-party imports, then local imports.
- Keep one import per line unless the file already uses grouped imports naturally.
- Follow the import style already present in the file rather than reformatting unrelated imports.
- Avoid introducing unused imports.

## Formatting Conventions

- Use `snake_case` for functions, local variables, and module-level helpers.
- Use `CamelCase` for classes.
- Use `UPPER_CASE` for constants, env var names, and top-level configuration constants.
- Keep line lengths reasonable, but do not rewrap large files unless needed for your change.
- Prefer explicit helper functions over dense inline logic in newer/refactored code.
- Preserve versioned filenames such as `*_v2.py` and `*_v6.py` unless renaming is requested.

## Type Hints and Data Structures

- Newer scripts in this repo use type hints, `dataclass`, and structured configuration objects.
- Older scripts often do not.
- When editing modern files, prefer to continue the typed style.
- When editing older scripts, add types only where they improve clarity and do not force broad churn.
- Favor dictionaries with stable keys only when the surrounding code already relies on them.
- Favor `dataclass` or small helpers for new structured config in refactored code.

## Naming Conventions

- Match existing domain terminology: `cancer`, `not_cancer`, `patient`, `split`, `manifest`, `artifact`, `annotation`.
- Preserve output folder names exactly when they are part of the pipeline contract.
- Do not silently rename `TRAIN`, `VALIDATION`, `TEST`, `CANCER`, `NOT_CANCER`, `CANCER_MASK`, or `NOT_CANCER_MASK`.

## Error Handling

- Fail fast on invalid required configuration or missing critical files.
- Use `ValueError` or `RuntimeError` for programming/configuration errors inside functions.
- Use `sys.exit(...)` in top-level script entrypoints when the file already follows that style.
- Use warnings or logged messages for partial-data problems when processing can safely continue.
- Preserve machine-readable output contracts, especially the JSON result emitted by `3_1_imageReader_refactored.py`.
- Do not swallow exceptions silently.

## Logging and Output

- Prefer `logging` for nontrivial workflows and long-running scripts.
- Many scripts configure root logging with both file and console handlers; follow that pattern when extending those files.
- Avoid excessive `print` in modern/refactored scripts unless the file already provides a terminal UX.
- In `2_database_manager.py`, styled terminal output with ANSI sequences and emoji is already part of the current UX; preserve that style if editing nearby code.

## Data Integrity Rules

- Preserve patient-level split isolation; avoid anything that can introduce leakage across `TRAIN`, `VALIDATION`, and `TEST`.
- Preserve image/mask pairing by filename.
- Preserve manifest schema and run metadata unless the user explicitly requests schema changes.
- Preserve label semantics: cancer remains positive class, not-cancer remains negative class.
- Do not change patch acceptance thresholds, stain normalization behavior, or artifact filtering semantics without clear intent.

## Filesystem and Artifact Safety

- This repo produces many large/generated outputs: PNG patches, logs, manifests, entropy caches, and HDF5 files.
- Avoid committing generated data unless the user explicitly asks.
- Treat `.env`, `database.db`, `DOWNLOAD_DRIVE/credentials.json`, `credentials.json`, and `token.json` as sensitive or local-state files.
- Respect `.gitignore`, which already ignores `.env`, logs, PNGs, bytecode, and credential files.
- Avoid deleting source folders or generated datasets unless the user explicitly requests destructive cleanup.

## Platform Assumptions

- The repo strongly assumes Windows in many places.
- `OPENSLIDE_PATH` and `os.add_dll_directory(...)` are used for OpenSlide setup.
- Hardcoded Windows drive paths are common in `__main__` blocks.
- If you are working from Linux or WSL, do not “fix” platform assumptions globally unless asked.

## Agent-Specific Guidance

- Prefer minimal, local edits over broad rewrites.
- First understand whether a file is legacy, helper, or actively used in the main pipeline.
- Do not introduce a new framework or project-wide tooling without an explicit request.
- If adding a new command to documentation, make sure it is actually supported by the repository as-is.
- When proposing validation, clearly distinguish between official repo workflows and ad hoc checks.
- If you touch scientific logic, be conservative and preserve reproducibility.
- If you touch split generation or sanity checks, assume correctness matters more than cleverness.

## Git / Workspace Notes

- In some environments, `git status` may fail with a dubious ownership warning.
- Do not change global git configuration unless the user explicitly asks.
- The worktree may contain local data files and generated artifacts; avoid treating them as safe to remove.

## Quick Summary

- This is a script-first research pipeline.
- There is no formal lint/test harness in-repo.
- The closest thing to a single test is running `3_1_imageReader_refactored.py` on one sample.
- Preserve data contracts, patient isolation, and filesystem conventions.
- Be careful with `.env`, Windows paths, OpenSlide setup, and large generated outputs.
