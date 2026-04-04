# Stage 4 Optimization Sampling Performance Findings

## Scope

- Target script: `4_1_optimization_sampling.py`
- Real benchmark source: `/mnt/host_f/SOURCE_DATASET.h5`
- Goal: identify whether Stage 4.1 has major speedup opportunities and measure before/after on the real HDF5 source without attempting a full 55k-row render.

## Source Dataset Snapshot

- Rows: `55,560`
- `images` shape: `(55560, 224, 224, 3)` `uint8`
- `masks` shape: `(55560, 224, 224)` `uint8`
- Label counts: `{1: 27292, 0: 28268}`
- Unique `patient_ids`: `96`

## What Was Measured

Measurements were taken on controlled subsets of the real source HDF5 using `time.perf_counter()` and `cProfile`.

The following paths were benchmarked:

- Stage 4 metadata discovery: `discover_hdf5_image_mask_pairs()`
- Stage 4 sample selection: `select_sample_stems()`
- Stage 4 overlay generation single-process and multiprocessing
- Isolated HDF5 image+mask row reads to separate read cost from contour drawing and PNG writes

## Baseline Findings

### 1. The entrypoint is not the bottleneck

`4_1_optimization_sampling.py` is a thin orchestrator. Runtime is dominated by helper code, especially overlay generation in `helpers/optimization_sampling/overlay.py`.

### 2. HDF5 row reads dominate overlay generation

Baseline `cProfile` on `20` single-process overlays:

- Total: `96.076s`
- `h5py` dataset `__getitem__`: `95.345s`
- `cv2.imwrite`: `0.127s`

Conclusion: the hot path is overwhelmingly HDF5 row access, not contour drawing or PNG encoding.

### 3. Filename discovery had avoidable scalar-read overhead

Real-source benchmark for the `filenames` dataset:

- Row-by-row scalar reads: `8.63s`
- Bulk `filenames[:]` read: `0.32s`

The original Stage 4 discovery loop was paying a large HDF5 scalar-read penalty up front.

## Implemented Changes

### Change 1. Bulk-read `filenames` during HDF5 discovery

File: `helpers/optimization_sampling/sampling.py`

- Replaced row-by-row `filenames[index]` reads with one bulk `filenames[:]` read and a Python loop over the in-memory values.

Expected benefit:

- materially lower Stage 4 startup time on large HDF5 inputs

### Change 2. Reuse one HDF5 handle per worker and read image+mask from the shared handle

File: `helpers/optimization_sampling/overlay.py`

- Added worker-local cached HDF5 handles
- Added a combined HDF5 image+mask read path when both refs point to the same source row
- Added cleanup for cached worker handles
- Added explicit `chunksize` to `Pool.imap_unordered(...)`

Expected benefit:

- reduce repeated HDF5 open/close overhead
- improve the raw image+mask read path in overlay generation

### Change 3. Order HDF5-backed overlay tasks by source row before dispatch

File: `helpers/optimization_sampling/overlay.py`

- Sort HDF5-backed `OverlayTask` items by source path and row index before passing them to the worker pool.
- Keep non-HDF5 tasks supported by ordering them after HDF5-backed tasks.

Expected benefit:

- reduce random-access churn inside each worker
- improve cache locality for nearby HDF5 rows

### Validation

- `uv run pytest tests/test_optimization_sampling_pipeline.py tests/test_optimization_sampling_overlay.py tests/test_optimization_sampling_config.py`
- `uv run ruff check helpers/optimization_sampling tests/test_optimization_sampling_pipeline.py tests/test_optimization_sampling_overlay.py tests/test_optimization_sampling_config.py`
- `uv run mypy helpers/optimization_sampling tests/test_optimization_sampling_pipeline.py tests/test_optimization_sampling_overlay.py tests/test_optimization_sampling_config.py`

All passed.

## Before / After Results

### Discovery and sampling

Before:

- `discover_hdf5_image_mask_pairs()`: `13.83s`
- `select_sample_stems()`: `0.056s`

After:

- `discover_hdf5_image_mask_pairs()`: `1.63s`
- `select_sample_stems()`: `0.058s`

Interpretation:

- discovery improved by about `8.5x`
- sample selection was never a meaningful bottleneck

### Isolated HDF5 read path

Benchmark: `100` image+mask row pairs from the real source HDF5, no overlay drawing, no PNG writes.

Before:

- `473.06s`

After:

- `331.17s`

Interpretation:

- raw HDF5 read path improved by about `30%`
- this confirms the handle reuse change is real and not just theoretical

### Single-process overlay profile

`cProfile` on `20` overlays:

Before:

- total: `96.08s`
- `h5py` dataset `__getitem__`: `95.35s`

After:

- total: `64.27s`
- `h5py` dataset `__getitem__`: `63.99s`

Interpretation:

- about `33%` faster on this profiled subset
- HDF5 row access remains the overwhelming bottleneck even after optimization

### End-to-end overlay generation

Same-subset single-process comparison on `40` overlays with output hashing:

- before: `197.77s`
- after: `208.81s`
- matching output hashes: `40/40`

Interpretation:

- correctness was preserved exactly on the compared outputs
- this single run did not show an end-to-end speedup; wall-clock noise from storage and output writes appears to dominate at this scale

Multiprocess comparison on `80` overlays with `4` processes:

- before: `388.37s`
- after: `378.60s`

Interpretation:

- about `2.5%` faster end-to-end on this subset
- useful, but not a massive improvement

Task-ordering comparison on `80` overlays with `4` processes:

- unsorted dispatch: `245.33s`
- sorted-by-row dispatch: `239.72s`

Interpretation:

- about `2.3%` faster on this direct ordering comparison
- small but consistent with the hypothesis that access ordering helps

Cached read-only comparison on `60` image+mask row pairs:

- unsorted reads: `187.31s`
- sorted reads: `174.29s`

Interpretation:

- about `6.9%` faster
- supports keeping the row-ordering change in production because the benefit appears in the underlying read path, not just one noisy end-to-end run

Post-change confirmation run on `80` overlays with `4` processes:

- ordered production path: `167.99s`

Interpretation:

- materially faster than earlier unordered runs on the same subset size
- this result is encouraging, but it should still be treated cautiously because filesystem cache state and HDF5 caching can move wall-clock times noticeably between runs

Optimized process-count sweep on `40` overlays:

- `1` process: `134.02s`
- `2` processes: `95.85s`
- `4` processes: `90.26s`

Interpretation:

- moderate benefit from parallelism exists on this workload
- gains are limited compared with the cost of random HDF5 reads

## Conclusions

### What improved materially

1. Stage 4 startup became much faster.
2. The raw HDF5 overlay read path became meaningfully faster.
3. Row-ordered task dispatch gives an additional modest improvement on top of handle reuse.
4. Overlay correctness was preserved on before/after hash checks.

### What did not improve massively

1. End-to-end overlay generation is still slow.
2. The dominant remaining cost is random HDF5 row access itself.
3. The current optimization removes open/close overhead, but it does not solve the much larger cost of `handle[dataset][row_index]` on this large HDF5 layout.

## Practical Answer

There was no evidence that `4_1_optimization_sampling.py` itself had a massive speedup opportunity.

The realistic optimization targets were inside helper code, and two safe improvements were implemented:

- faster HDF5 filename discovery
- cached worker-local HDF5 access for overlay generation
- row-ordered HDF5 task dispatch for better locality

These changes help, especially the startup path and isolated read path, but they do not produce a massive end-to-end runtime collapse on the current source HDF5.

## Highest-Value Remaining Opportunities

If more Stage 4 speed is needed, the next candidates should be benchmarked before implementation:

1. Batch or reorder selected row reads by HDF5 row index instead of fully random access.
2. Benchmark whether a different HDF5 chunk/cache strategy materially helps this exact dataset.
3. Compare HDF5-backed overlay generation versus reading from original image/mask files if the Stage 3 source also retains fast local source paths.
4. Benchmark larger subsets across `1`, `2`, and `4` workers before changing the default `OPTIMIZATION_SAMPLING_NUM_PROCESSES`.

At this point, the biggest remaining gains are likely to require changing the access pattern, not just micro-optimizing the current one-row-at-a-time HDF5 reads.
