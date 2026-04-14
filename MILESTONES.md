# Stage 5 Crossfold Simplification Milestones

## Milestone 1 --DONE: Replace Stage 5 Split Config Surface
- Remove the legacy Stage 5 split env vars and config fields that no longer match the target algorithm.
- Keep `CROSSFOLD_RANDOM_STATE`.
- Add explicit patient-capacity config for `CROSSFOLD_TEST_PATIENT_COUNT` and `CROSSFOLD_VALIDATION_PATIENT_COUNT`.
- Add `CROSSFOLD_SPLIT_OPTUNA_TRIALS` for the Optuna search budget.
- Remove ratio, adaptive sizing, entropy-objective split tuning, class-balance heuristics, train-image-dominance, and validation-sufficiency config.
- Update `.env`-facing documentation and config parsing tests to match the new surface.

## Milestone 2 --DONE: Rebuild Patient Profiling For The New Algorithm
- Update `helpers/crossfold/splitting.py` patient aggregation to produce patient metadata with:
  - `patient_id`
  - `total_samples`
  - `cancer_samples`
  - `non_cancer_samples`
- Compute the global sample-level cancer ratio once from the dataset.
- Preserve deterministic patient ordering before optimization so seeded runs stay reproducible.

## Milestone 3 --DONE: Implement Deterministic Greedy Allocation
- Implement a greedy allocator that assigns patients into `TEST`, `VALIDATION`, and `TRAIN` based on the smallest hypothetical delta from the global cancer ratio.
- Respect exact patient capacities for `TEST` and `VALIDATION`.
- Assign all remaining patients to `TRAIN`.
- Use the agreed tie-break priority when deltas are equal:
  - `TEST`
  - `VALIDATION`
  - `TRAIN`
- Add focused unit tests for allocation behavior and tie-break handling.

## Milestone 4 --DONE: Implement Optuna-Based Split Optimization
- Add an Optuna objective that suggests one continuous weight per patient.
- Sort patients by those suggested weights.
- Run the greedy allocator for each trial.
- Minimize the final loss:
  - `abs(train_ratio - global_ratio) + abs(val_ratio - global_ratio) + abs(test_ratio - global_ratio)`
- Use `TPESampler` with the configured random seed for deterministic reproducibility.
- Re-run the greedy allocator once using the best trial weights to build the final split.

## Milestone 5 --DONE: Apply The Final Guardrails
- Validate early that `TEST` patient count is at least `20`.
- Validate early that `VALIDATION` patient count is at least `20`.
- Validate early that `TEST + VALIDATION < total_patients` so `TRAIN` remains non-empty.
- Fail fast after final split generation if `TEST` has only one class.
- Fail fast after final split generation if `VALIDATION` has only one class.
- Remove legacy Stage 11 validation-sufficiency, adaptive sizing, and other non-algorithm guardrails.

## Milestone 6 --DONE: Add Pure NumPy Statistical Verification
- Compute per-split observed cancer and non-cancer sample counts.
- Compute expected counts from the global cancer ratio.
- Compute raw chi-square statistics with NumPy only:
  - `np.sum((observed - expected) ** 2 / expected)`
- Record per-split verification values and the final optimization loss for auditability.
- Add tests for verification metric calculation.

## Milestone 7 --DONE: Integrate The New Splitter Into The Stage 5 Pipeline
- Replace the existing Stage 5 split-selection call in `helpers/crossfold/pipeline.py` with the new Optuna + greedy splitter.
- Remove entropy-driven split optimization wiring.
- Ensure entropy is computed only when normalization needs it.
- Keep normalization behavior untouched after split generation.
- Keep patient-level isolation and downstream pipeline contracts intact.

## Milestone 8 --DONE: Preserve Existing Stage 5 Performance Wins
- Leave `helpers/crossfold/io.py` batched HDF5 copy/write paths unchanged.
- Leave provenance reuse and bulk verification behavior unchanged.
- Keep manifest generation and split writing flow intact aside from new split metadata.
- Ensure the new optimization runs only on compact patient-level metadata, not on sample-level or HDF5-heavy paths.
- Benchmark split-selection separately from downstream write/verify steps if runtime risk appears.

## Milestone 9 --DONE: Update Metadata, Outputs, And Benchmarks
- Update Stage 5 metadata to describe the new split-selection method.
- Remove legacy `neutral_stratified`, entropy-objective, and optimization metadata that no longer applies.
- Record Optuna trial count, global cancer ratio, final loss, and per-split chi-square statistics in run metadata.
- Update `scripts/benchmark_stage5_crossfold.py` to compare and measure the new splitter appropriately.

## Milestone 10 --DONE: Replace Legacy Tests With Algorithm-Focused Coverage
- Update `tests/test_crossfold_config.py` for the new config surface.
- Replace legacy splitting tests that depend on ratio sizing, `sklearn`, entropy-guided selection, or Stage 11 validation sizing.
- Add tests for:
  - exact `TEST` and `VALIDATION` capacities
  - deterministic seed behavior
  - tie-break priority
  - final class guardrails
  - loss computation
  - NumPy chi-square verification
  - unchanged pipeline behavior after split generation
