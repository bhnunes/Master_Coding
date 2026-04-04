# Stage 4.3 Graph Cleaning Performance Findings

## Scope

This document records a sampled before/after performance investigation for `4_3_cleaner_script.py` and `helpers/graph/cleaning_pipeline.py`.

The Stage 4.3 entry script itself is thin orchestration. The real runtime is spent in candidate discovery and per-row contamination scoring.

## Benchmark Inputs

- Parameter artifact: `/mnt/host_f/graph_cleaning_params.json`
- Source dataset: `/mnt/host_f/SOURCE_DATASET.h5`
- Source dataset shape:
  - `images`: `(55560, 224, 224, 3)`
  - `masks`: `(55560, 224, 224)`
- Sample sizes used for scoring benchmarks: `64` and `128` leading rows

The full source dataset was not benchmarked end-to-end because the user requested sampled measurements only.

## Baseline Findings

### Candidate discovery hotspot

Baseline `_list_hdf5_candidates(...)` was slow for two avoidable reasons:

1. It computed a full-file SHA256 of the source HDF5 on every run.
2. It rebuilt image/mask refs through `discover_hdf5_image_mask_pairs(...)` and then reopened the HDF5 to iterate row-by-row.

Measured candidate discovery times on the sample source HDF5:

- baseline listing: `158.830s`
- optimized listing: `0.870s`
- speedup: `182.7x`

Subcomponent measurements from the original path:

- `discover_hdf5_image_mask_pairs(...)`: about `0.844s`
- `hash_file_sha256(...)`: about `189.180s`
- direct bulk metadata read (`filenames`, `patient_ids`, `slide_ids`): about `0.904s`

Interpretation: the dominant startup cost was hashing the entire HDF5 instead of reusing the existing `source_signature` attribute.

### Scoring hotspot

`cProfile` on the original sampled processing path showed that the main bottleneck was not graph segmentation. It was repeated row-at-a-time HDF5 reads.

For `64` sampled rows, baseline profile highlights were:

- `h5py._hl.dataset.__getitem__`: about `405.4s`
- `skimage.segmentation._felzenszwalb.felzenszwalb`: about `7.0s`

Interpretation: Stage 4.3 was paying a very large penalty for single-row HDF5 access.

## HDF5 Access Pattern Investigation

Measured image+mask read time for the first `64` rows using different read strategies:

- row-by-row (`batch=1`): `240.585s`
- batches of `8`: `22.496s`
- batches of `32`: `5.500s`
- batch of `64`: `3.109s`

Interpretation: contiguous batched reads are dramatically faster than repeated single-row reads on this HDF5 layout.

## Implemented Changes

### 1. Removed full-file hashing from the hot path when possible

`_list_hdf5_candidates(...)` now:

- reads `source_signature` from the HDF5 attrs when present,
- falls back to `hash_file_sha256(...)` only if the attribute is missing,
- reads `filenames`, `patient_ids`, and optional `slide_ids` in one pass,
- builds `::images[index]` and `::masks[index]` refs directly without `discover_hdf5_image_mask_pairs(...)`.

### 2. Added batched HDF5 loading for Stage 4.3 scoring

When Stage 4.3 is using the default contamination scorer over contiguous HDF5-backed candidates, `_process_hdf5_candidates(...)` now:

- reads contiguous image/mask slices in batches of `64`,
- reuses those loaded arrays for per-image decisions,
- calls `calculate_roi_contamination_from_arrays(...)` instead of reopening HDF5 rows one-by-one,
- preserves decision ordering and output behavior.

### 3. Kept manifest writing unchanged

Manifest writing was already negligible compared with scoring and did not require optimization.

## After Results

Measured sampled scoring results after the batching change:

- `64` rows:
  - baseline: `185.543s`
  - optimized: `9.178s`
  - speedup: `20.2x`
  - rejected rows matched: `26`
- `128` rows:
  - baseline: `364.343s`
  - optimized: `14.482s`
  - speedup: `25.2x`
  - rejected rows matched: `31`

Manifest-writing overhead for `128` sampled rows remained tiny:

- processing: `21.360s`
- writing both manifests: `0.015s`

The processing number is still much larger than manifest output, confirming that scoring remains the real runtime cost after the HDF5 fixes.

## Post-Change Profile

`cProfile` on the optimized `64`-row sampled path showed the hotspot balance changed substantially:

- `felzenszwalb`: about `4.71s`
- batched `h5py.__getitem__`: about `4.41s`

Interpretation: after fixing the worst HDF5 access pattern, segmentation and batched HDF5 reads are now of the same order of magnitude.

## Correctness Validation

On the sampled rows benchmarked above, accepted/rejected decisions matched between baseline and optimized paths.

Additional regression coverage was added for:

- reusing `source_signature` instead of hashing when available,
- batching contiguous HDF5 reads in Stage 4.3 processing.

## Remaining Opportunities

Follow-up exploration after the first optimization pass compared larger contiguous HDF5 batches against multiprocessing.

### Larger batch-size tuning

Measured sampled scoring times after the initial batched-read rewrite:

- `256` rows:
  - batch `32`: `36.614s`
  - batch `64`: `23.710s`
  - batch `128`: `17.032s`
  - batch `256`: `12.967s`
- `512` rows:
  - batch `64`: `37.248s`
  - batch `128`: `22.669s`
  - batch `256`: `17.407s`
  - batch `512`: `14.369s`

Interpretation: larger contiguous HDF5 batches remained beneficial on the sampled workload. Stage 4.3 now uses a larger default batch size of `512` rows.

### Multiprocessing prototype

A sampled `ProcessPoolExecutor` prototype over contiguous batches was also measured on this 4-core machine:

- `128` rows:
  - serial batched: `16.829s`
  - `2` workers: `12.259s` (`1.37x`)
  - `4` workers: `12.112s` (`1.39x`)
- `256` rows:
  - serial batched: `23.020s`
  - `2` workers: `16.514s` (`1.39x`)
  - `4` workers: `16.404s` (`1.40x`)

Interpretation: multiprocessing helped, but much less than simply increasing the contiguous batch size. On this workload, larger slice reads delivered the better complexity/performance tradeoff.

The following opportunities remain, but they were not needed for the large speedups already achieved:

1. `num_workers` is still only logged and not used by Stage 4.3. Parallel CPU-side scoring could reduce runtime further once the HDF5 access pattern is no longer dominating.
2. If Stage 4.3 still needs more speed, the next step should be to benchmark multiprocessing over already-batched array chunks, not row-at-a-time HDF5 access.
3. If future datasets do not provide `source_signature`, the fallback SHA256 path will still be expensive by design.

## Conclusion

The biggest Stage 4.3 wins came from fixing data access, not from changing the graph contamination algorithm itself.

The two highest-impact changes were:

1. reuse the HDF5 `source_signature` instead of hashing the entire file during candidate discovery,
2. replace row-at-a-time HDF5 reads with contiguous batched reads during scoring.

These changes reduced sampled Stage 4.3 setup time by about `182x` and sampled scoring time by about `20x` to `25x` while preserving decisions on the measured rows.
