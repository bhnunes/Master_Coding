# Stage 4.2 Graph Tuning Performance Findings

## Scope

- Target: `4_2_tune_graph_method.py`
- Focus: Stage 4.2 Bayesian graph tuning runtime after switching to `skimage.segmentation.felzenszwalb`
- Date: 2026-04-04

## Workload Used

- Available review samples under `/mnt/host_f/APPROVED` and `/mnt/host_f/REJECTED`
- Sample counts discovered: `270` approved and `115` rejected (`385` total)
- Proxy benchmark subset used for before/after timing: first `80` approved and first `40` rejected PNGs (`120` total)
- PNG size: `224x224`

## Limitation

The configured Stage 4.2 source HDF5 path from `.env` was not present in this workspace during the benchmark pass, so exact end-to-end HDF5-backed timing could not be reproduced here.

The benchmark therefore used the available reviewed PNGs as a proxy workload to measure the two main costs that still matter for Stage 4.2:

- graph segmentation and contamination scoring cost per image
- repeated rescoring across grouped cross-validation folds inside the Bayesian objective

## Root Cause

The dominant avoidable cost was not individual image scoring itself. It was repeated scoring of the same reviewed records within a single Bayesian objective evaluation.

Before this change, one objective evaluation did this for every fold:

- score training fold records
- fit threshold
- score validation fold records

Since contamination rate depends only on `(record, params)`, the same record was rescored once per fold even though the score did not change between folds.

With the current default config and the observed log split (`308` training records, `3` inner folds, `50` Bayesian calls), that pattern multiplies scorer calls by about `3x`.

## Implemented Changes

### `helpers/graph/tuning_pipeline.py`

- Added one-time reviewed record preloading for the default contamination scorer
- Changed `_build_objective()` to score each record once per parameter set and reuse those scores across all CV folds
- Reused preloaded image/mask arrays during final train/test threshold evaluation
- Added regression coverage that locks in one scorer call per record per objective evaluation

### `helpers/graph/contamination.py`

- Added a combined source reader for image/mask pairs
- Added cached HDF5 handles instead of reopening the same HDF5 file for each read
- Added `calculate_roi_contamination_from_arrays()` so Stage 4.2 can work directly from preloaded reviewed data
- Replaced per-segment Python masking loops with vectorized `np.bincount` aggregation for segment mean intensity classification

## Benchmarks

### Review Label Discovery

- `collect_review_labels(Path('/mnt/host_f'))`: `1.7695s` for `385` labels

### Scoring Microbenchmark

Proxy setup:

- `120` review PNGs loaded in memory
- full-image ROI mask used for consistent per-image scoring comparison
- same graph parameters for both runs: `bg_intensity_thresh=198`, `k=386`, `min_size=200`, `erosion_px=0`

Results:

- baseline scorer emulation: `14.1807s`
- optimized scorer path: `14.1561s`
- speedup: `1.002x`
- checksum match: yes (`16.718371` both)

Interpretation:

- once the images are already in memory, segmentation itself dominates
- the vectorized post-segmentation path is correct, but the big runtime win does not come from that layer alone

### Objective Evaluation Benchmark

Same `120` review PNG proxy workload with grouped CV (`3` folds):

- baseline fold-by-fold rescoring objective: `42.2590s`
- optimized once-per-record-per-params objective: `15.3468s`
- speedup: `2.754x`
- objective value match: yes (`-0.666667` both)

## Conclusion

The confirmed high-impact optimization for Stage 4.2 is eliminating duplicate scorer work inside the Bayesian objective.

Measured on the available review proxy set:

- objective runtime improved by about `2.75x`
- objective output stayed unchanged for the benchmarked parameter set

The remaining runtime hotspot is the segmentation call itself. Additional gains are possible, but they are less likely to be "massive" than the now-implemented objective-level deduplication.

## Recommended Next Steps

1. Re-run `4_2_tune_graph_method.py` in the target environment and compare wall-clock runtime against the previous run.
2. If the source HDF5 is available there, benchmark one full Stage 4.2 run with the current code and append the real end-to-end timing here.
3. If Stage 4.2 is still too slow, the next optimization candidate is batched multiprocessing of record scoring per Bayesian objective, but only after measuring serialization overhead on the real HDF5-backed workload.
