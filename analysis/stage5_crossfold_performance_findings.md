# Stage 5 Crossfold Performance Findings

## Scope

This document analyzes Stage 5 performance for `5_crossfold.py` and the helper package under `helpers/crossfold/`.

Goal:
- identify the likely dominant runtime costs
- estimate algorithmic complexity
- propose high-impact, low-risk optimizations before implementation

This document started as a theoretical and code-structure analysis. A first implementation pass is now in place, but this file still does not claim measured Stage 5 speedups yet.

## Implemented First Pass

The following Stage 5 changes have now been implemented:

- cached source-HDF5 provenance reuse in `helpers/crossfold/pipeline.py`, `helpers/crossfold/io.py`, and `helpers/crossfold/provenance.py`
- batched HDF5-backed entropy reads in `helpers/crossfold/entropy.py`
- configurable Stage 5 split-output compression via `CROSSFOLD_HDF5_COMPRESSION`
- configurable Stage 5 split-copy batching via `CROSSFOLD_COPY_BATCH_SIZE`
- batched split writing in `helpers/crossfold/io.py`
- reduced split-search pandas rescans in `helpers/crossfold/splitting.py` through cached patient metadata and array-based objective scoring

Validation completed for this first pass:

- targeted crossfold tests passed
- Ruff passed on touched crossfold files
- MyPy passed on touched crossfold files

## Entry Point Reality

`5_crossfold.py` is only a thin entrypoint:

- loads `.env`
- builds `CrossfoldConfig`
- calls `run_crossfold_pipeline(config)`

The actual runtime lives in:

- `helpers/crossfold/discovery.py`
- `helpers/crossfold/entropy.py`
- `helpers/crossfold/splitting.py`
- `helpers/crossfold/io.py`
- `helpers/crossfold/provenance.py`
- `helpers/crossfold/normalization.py`

## Current Stage 5 Flow

1. Load source HDF5 metadata into a dataframe.
2. If the entropy objective is enabled, compute patch entropy for every row.
3. Aggregate patch entropy to patient-level medians.
4. Search randomized patient-level TRAIN/VALIDATION/TEST splits.
5. Optionally fit stain normalization on TRAIN.
6. Write `TRAIN.h5`, `VALIDATION.h5`, and `TEST.h5`.
7. Verify written HDF5 integrity.
8. Write manifest and provenance artifacts.

## Complexity Overview

Let:

- `N` = number of source patches
- `P` = number of patients
- `T` = number of split attempts (`max_tries`)
- `S` = size of the source HDF5 file in bytes

Approximate current costs:

- source metadata load: `O(N)`
- entropy computation: effectively `O(N * (file_open + row_read + resize + histogram))`
- patient entropy aggregation: `O(N)`
- split search: roughly `O(T * P)` plus repeated pandas filtering overhead
- split writing: `O(N * row_read_write)` plus output compression cost
- provenance hashing: effectively `O(k * S)` where `k` is the number of times the same source HDF5 is rehashed in one run

The current architecture strongly suggests the dominant practical cost is HDF5 I/O, not the split math itself.

## Likely Bottlenecks Ranked

### 1. Per-row HDF5 reopen in entropy computation

Code:

- `helpers/crossfold/entropy.py:54-64`
- `helpers/crossfold/entropy.py:67-110`

Current behavior:

- each entropy task reopens the same HDF5 file
- one row is read
- the file is closed again

Why this is expensive:

- repeated file-open overhead per patch
- repeated HDF5 metadata lookup per patch
- poor read locality
- multiprocessing uses `spawn`, which adds additional worker startup overhead

Why this likely explains the reported runtime:

- even for only 10 patients, `N` can still be very large
- runtime follows patch count, not patient count
- if each patch causes a new HDF5 open, Stage 5 can become catastrophically slow

### 2. Repeated hashing of the full source HDF5

Code:

- `helpers/crossfold/io.py:20-36`
- `helpers/crossfold/io.py:71-90`
- `helpers/crossfold/provenance.py:154-207`
- `helpers/provenance.py:62-80`

Current behavior:

- Stage 5 computes SHA256 over the full source HDF5 multiple times in the same run

Likely repeats:

- once in each split signature build
- once again while writing each split file attribute
- once again when building run provenance

Why this is expensive:

- each full-file hash is `O(S)`
- on large Stage 3 outputs, this can add a large fixed tax
- this is avoidable because the source file does not change during the run

### 3. Row-by-row split writing with hardcoded gzip compression

Code:

- `helpers/crossfold/io.py:59-136`

Current behavior:

- each source row is read separately
- each destination row is written separately
- output `images` and `masks` datasets are always created with `compression="gzip"`

Why this is expensive:

- scalar HDF5 reads/writes are much slower than batched reads/writes
- gzip adds substantial CPU cost
- when normalization is disabled, this path still pays row-by-row overhead unnecessarily

### 4. Split search performs repeated pandas scans inside every attempt

Code:

- `helpers/crossfold/splitting.py:184-187`
- `helpers/crossfold/splitting.py:244-250`
- `helpers/crossfold/splitting.py:259-274`
- `helpers/crossfold/entropy.py:123-132`

Current behavior:

- each attempt repeatedly filters `patient_df`
- image counts, class coverage, validation support, and entropy scoring all rescan pandas structures

Why this matters:

- not likely the first bottleneck for the reported runtime
- still becomes important when `max_tries` is large or feasible splits are rare

### 5. Full output reread for integrity verification

Code:

- `helpers/crossfold/io.py:139-164`

Current behavior:

- after writing each split HDF5, Stage 5 rereads all metadata rows to confirm alignment

Why this matters:

- adds another full pass over every output split
- likely secondary compared with entropy and write-path costs
- may still be meaningful on large datasets

## Detailed Code Findings

### Entropy path

Current entropy pipeline:

- `compute_all_patch_entropies()` builds a work list for every patch
- HDF5-backed rows use `calculate_image_entropy_from_hdf5_row()`
- each call does `with h5py.File(..., "r")`
- multiprocessing uses `spawn`

Structural issues:

- file handle reuse is absent
- batching is absent
- contiguous read coalescing is absent
- results are materialized patch-by-patch into pandas before patient reduction

Expected outcome:

- this is the first optimization target

### Split search path

Current split search is conceptually fine, but it spends extra time in pandas:

- `image_count()` filters `patient_df` every time
- `split_has_both()` filters `patient_df` every time
- `_validation_supports_stage11()` filters `patient_df` every time
- entropy scoring filters `patient_entropy_df` every time

Algorithmically this is not terrible, but implementation is not cache-friendly.

### Write path

Current Stage 5 write path resembles the pre-optimization Stage 3 packaging path:

- one row read at a time
- one row write at a time
- compression always enabled
- no sorted source-row batching

Stage 3 already demonstrated that batched HDF5 read/write plus configurable compression can produce major gains.

### Provenance path

The provenance design is scientifically sound, but the implementation currently recomputes stable source-file information too many times.

The source HDF5 provenance should be computed once at pipeline start and reused everywhere downstream.

## Prioritized Optimization Plan

## 1. Cache source HDF5 provenance once per run

Change:

- compute `collect_hdf5_provenance(source_hdf5_path)` once in `run_crossfold_pipeline()`
- pass the cached payload into split writing and run-config generation

Expected impact:

- high fixed-cost reduction on large HDF5 inputs

Risk:

- low

Why first:

- simple
- low-risk
- easy to verify

## 2. Redesign entropy computation for HDF5-backed datasets

Change:

- open each HDF5 source once per worker, not once per row
- group work by `source_hdf5_path`
- sort row indices and process contiguous spans in batches
- compute entropies from in-memory arrays after batched reads

Expected impact:

- very high
- likely the largest Stage 5 win

Risk:

- medium

Notes:

- preserve exact entropy semantics
- keep path-based fallback for non-HDF5 rows
- add regression coverage for HDF5-backed rows

## 3. Add Stage 5 HDF5 compression control

Change:

- add `CROSSFOLD_HDF5_COMPRESSION` with allowed values `none`, `lzf`, `gzip`
- default to `none` unless evidence proves otherwise

Expected impact:

- high for split writing

Risk:

- low to medium

Notes:

- larger files are acceptable if runtime savings are large
- mirror the successful Stage 3 pattern

## 4. Rewrite split writing as batched HDF5 copy

Change:

- sort rows by `source_row_index`
- read source `images` and `masks` in contiguous slices
- write destination in batches
- keep output order stable using a remapping step if needed
- if normalization is disabled, use direct batch copy without per-row Python work

Expected impact:

- high

Risk:

- medium

Notes:

- this is likely the second-largest runtime win
- reuse lessons from `helpers/packaging/writer.py`

## 5. Replace repeated split-search dataframe scans with cached arrays/maps

Change:

- precompute patient metadata once:
  - `patient_id -> label`
  - `patient_id -> n_images`
  - `patient_id -> entropy_median`
- evaluate each candidate split from those precomputed structures

Expected impact:

- medium

Risk:

- low

Notes:

- this improves scalability for high `max_tries`
- likely not the first bottleneck to fix

## 6. Score entropy objective from arrays instead of pandas filtering

Change:

- replace `patient_entropy_df[patient_entropy_df["patient_id"].isin(...)]`
- use a patient-indexed NumPy array or dictionary lookup

Expected impact:

- medium

Risk:

- low

## 7. Reassess integrity verification strategy

Change options:

- keep current full verification in strict mode
- add a faster default mode and an explicit strict mode
- verify metadata during write rather than by full reread

Expected impact:

- medium

Risk:

- medium because this touches scientific safety checks

Recommendation:

- do not relax this until the larger I/O bottlenecks are fixed and measured first

## Benchmarks To Run Before and After Changes

Minimum benchmark set:

1. time `compute_all_patch_entropies()` alone on the real source HDF5
2. time one call to `collect_hdf5_provenance(source_hdf5_path)`
3. time `create_train_val_test_split_best()` with objective enabled and disabled
4. time `write_split_hdf5()` on one representative TRAIN split with normalization off
5. time full Stage 5 run with:
   - objective on, normalization off
   - objective off, normalization off

This will separate:

- entropy cost
- provenance hashing cost
- split-search cost
- split-write cost

## Recommended Implementation Order

1. cache source provenance once per run
2. optimize HDF5-backed entropy computation
3. add configurable Stage 5 HDF5 compression
4. optimize split HDF5 writing with batched reads/writes
5. optimize split-search internals
6. re-measure and decide whether integrity verification changes are needed

## Expected High-Level Outcome

If the current runtime report is accurate, the biggest practical wins should come from:

- removing per-row HDF5 opens during entropy
- removing repeated full-file hashing
- removing row-by-row gzip-heavy split writing

Those three changes alone should materially reduce Stage 5 runtime before touching more subtle optimization work.

## Implementation Guardrails

Any optimization work must preserve:

- patient-level split isolation
- stable row alignment
- filename-keyed provenance
- label semantics (`1` = cancer, `0` = not-cancer)
- TRAIN-only normalization fitting
- deterministic provenance outputs where currently required

## Measured Benchmarks

Benchmark dataset:

- source HDF5: `/mnt/host_f/CHILE_OUTPUT/PATCHES/SOURCE_DATASET.h5`
- source size: `1,445,446,696` bytes
- rows: `7176`
- patients: `10`
- normalization: `NOT_NORMALIZED`
- entropy thumbnail: `128`
- entropy chunksize / read batch size: `128`

Notes:

- the direct baseline-versus-current entropy comparison was measured on a `1024`-row subset because the old entropy implementation is extremely slow
- full optimized timings were measured on the full `7176`-row dataset
- benchmark commands were run in phases rather than as one monolithic script run because the all-in-one harness exceeded the shell timeout once the slow baseline paths were included

### Entropy Benchmarks

Measured with one worker for direct before/after comparison on the HDF5-backed entropy path.

- baseline entropy on `1024` rows: `629.833725s`
- current entropy on `1024` rows: `8.451611s`
- current entropy on full `7176` rows: `59.008838s`
- patient entropy aggregation on full `7176` rows: `0.038175s`

Observed impact:

- `1024`-row entropy speedup: about `74.5x`
- simple linear extrapolation of the old entropy path to the full `7176` rows gives about `4415s` (`73.6 min`)
- that extrapolated baseline is consistent with the prior user report that the entropy stage alone was taking about one hour on this 10-patient source dataset

Interpretation:

- the Stage 5 entropy bottleneck was real and dominant
- removing per-row HDF5 reopen overhead produced the largest confirmed improvement in this pass

### Provenance Benchmarks

- one `collect_hdf5_provenance(...)` call: `18.282125s`
- seven repeated calls: `135.249285s`

Observed impact:

- the old repeated-hashing pattern could spend about `117s` of avoidable extra time on this source file in one run
- caching source provenance once per Stage 5 run removes that repeated fixed cost

### Split Search Benchmarks

Measured with the entropy objective enabled and feasible `2/2/6` patient sizing for this 10-patient cohort.

- baseline split search: `5.603573s`
- current split search: `2.230689s`

Observed impact:

- split-search internals are about `2.5x` faster after replacing repeated pandas scans with cached patient metadata and array-based objective scoring

Interpretation:

- split search was not the main Stage 5 bottleneck
- the optimization is still worthwhile because it reduces overhead once entropy and write-path costs are improved

### Split Write Benchmarks

Random `256`-row subset benchmark:

- baseline write, hardcoded `gzip`: `274.020767s`
- current write, `CROSSFOLD_HDF5_COMPRESSION=none`: `214.324345s`
- current write, `CROSSFOLD_HDF5_COMPRESSION=gzip`: `262.340172s`

Observed impact on the `256`-row subset:

- baseline to current `none`: about `1.28x` faster
- current `none` versus current `gzip`: about `1.22x` faster

Full `7176`-row current write benchmark:

- current write, `none`: `106.934025s`
- current write, `gzip`: `301.640583s`

Observed impact on the full dataset:

- `none` is about `2.82x` faster than `gzip`

Interpretation:

- for Stage 5 writes, compression choice matters substantially on the real full dataset
- `CROSSFOLD_HDF5_COMPRESSION=none` is the current best-known runtime setting
- the subset benchmark shows a modest baseline-to-current improvement, but the full-dataset benchmark shows that avoiding `gzip` is a major win in practice

### Copy Batch Size Sweep

Measured on the real Stage 5 split outputs with `CROSSFOLD_HDF5_COMPRESSION=none` and objective-disabled split sizing (`TRAIN=4274`, `VALIDATION=809`, `TEST=2093`).

Total Stage 5 split write + verify time by batch size:

- `64`: `216.191923s`
- `128`: `131.544246s`
- `256`: `98.514987s`
- `512`: `86.248415s`
- `1024`: `76.917481s`

Write-only totals by batch size:

- `64`: `197.319194s`
- `128`: `115.917580s`
- `256`: `83.231058s`
- `512`: `69.420386s`
- `1024`: `62.342810s`

Verify-only totals by batch size:

- `64`: `18.872728s`
- `128`: `15.626666s`
- `256`: `15.283930s`
- `512`: `16.828028s`
- `1024`: `14.574671s`

Baseline versus optimized verification on the real `1024`-batch split outputs:

- baseline total verification: `22.530965s`
- current total verification: `0.044987s`

Observed impact:

- verification is about `501x` faster after replacing row-by-row metadata rereads with bulk array reads and vectorized comparisons

Interpretation:

- larger batch sizes clearly help Stage 5 on this workload
- `1024` is the current best measured Stage 5 copy batch size on the real 10-patient dataset
- verification used to be a meaningful secondary cost, but the bulk-verification rewrite reduced it to a negligible share of Stage 5 runtime

### Full Stage 5 Runtime Benchmarks

Full optimized Stage 5 run on the real 10-patient dataset, using a safe benchmark symlink workspace and `CROSSFOLD_HDF5_COMPRESSION=none`.

- full run, objective enabled: `253.111787s`
- full run, objective disabled: `157.871166s`

Observed impact:

- optimized full Stage 5 runtime with the entropy objective enabled is about `4.22 min`
- optimized full Stage 5 runtime with the entropy objective disabled is about `2.63 min`
- objective-on versus objective-off adds about `95.24s`, which matches the now-much-cheaper entropy/objective overhead

Full Stage 5 runtime with larger copy batches:

- full run, objective enabled, batch `1024`: `189.662687s`
- full run, objective disabled, batch `1024`: `128.207308s`

Full Stage 5 runtime after the bulk-verification rewrite:

- full run, objective enabled, batch `1024`: `193.054036s` on a verification-optimized rerun
- full run, objective disabled, batch `1024`: `114.358549s`

Stage 5 progress-logging regression check:

- full run, objective disabled, batch `1024`, with progress logging enabled: `99.335629s`

Observed impact versus batch `256`:

- objective enabled: `253.111787s` -> `189.662687s` (`1.33x` faster)
- objective disabled: `157.871166s` -> `128.207308s` (`1.23x` faster)

Observed impact versus the earlier batch-`1024` build before the verification rewrite:

- objective enabled: roughly flat within runtime noise (`189.7s` versus `193.1s` across reruns)
- objective disabled: `128.207308s` -> `114.358549s` (`1.12x` faster)

Interpretation:

- after the first optimization pass, the full Stage 5 run is now minutes rather than approximately an hour-scale entropy phase by itself
- the entropy bottleneck has been reduced enough that write/verify/provenance work is now a much larger share of total runtime
- increasing the Stage 5 copy batch size to `1024` produced an additional end-to-end gain on the real dataset
- once verification was vectorized, write time remained the main Stage 5 post-entropy cost
- the lightweight progress logging added for entropy and split writing did not show evidence of meaningful runtime regression in the measured objective-off rerun

## Current Best-Known Stage 5 Runtime Settings

- `CROSSFOLD_HDF5_COMPRESSION=none`
- `CROSSFOLD_COPY_BATCH_SIZE=1024`
- `CROSSFOLD_NORMALIZATION_METHOD=NOT_NORMALIZED` when normalization is intentionally disabled

## Remaining Opportunities

After the first measured optimization pass, the biggest remaining Stage 5 runtime opportunities are now:

1. reduce HDF5 write and verification cost further, especially for large split outputs
2. benchmark whether batch sizes above `1024` still help or have plateaued on the full real dataset
3. consider lightweight Stage 5 progress logging for entropy and split writing, following the Stage 3 model
4. if desired, profile remaining write-path hotspots inside `helpers/crossfold/io.py` now that verification cost is near-zero

## Status

Current status: first optimization pass plus bulk verification rewrite implemented and benchmarked on the real 10-patient Stage 5 source dataset.

## Current User-Visible Behavior

Stage 5 now emits lightweight log-based progress for:

- HDF5-backed entropy computation
- split HDF5 writing
- split verification start markers

The progress logs include processed rows, throughput, remaining rows, and ETA, following the same general observability pattern used earlier in Stage 3.
