# Stage 3 Packaging Performance Findings

## Scope

- Target: `3_pack_splits_to_hdf5.py` merge and filtering path.
- Real bottleneck location: `helpers/packaging/writer.py`.
- Goal: reduce Stage 3 runtime substantially while preserving dataset contracts and deterministic ordering.

## What Was Slow

- The Stage 3 entrypoint itself was not expensive.
- The merge path in `helpers/packaging/writer.py` was doing three costly things:
  - reading every row into Python objects before writing anything
  - writing HDF5 datasets one row at a time
  - recomputing per-image and per-mask SHA256 digests during merge provenance
- Stage 3 also hardcoded `gzip` compression for output `images` and `masks`.

## Baseline Evidence

### Baseline benchmark input

- Real shard input: `/workspace/analysis/stage3_benchmark_single_shard`
- Backed by real Stage 2 shard:
  - `HHHA-2016-00083 B1 DGTE.h5`
  - rows: `786`

### Baseline timing

Command used:

```bash
uv run python -c "import os, time; os.environ['PACKAGING_SOURCE_HDF5_PATH']='/workspace/analysis/stage3_benchmark_single_shard'; os.environ['PACKAGING_OUTPUT_DIR']='/workspace/analysis/stage3_benchmark_baseline_single'; os.environ['PACKAGING_OUTPUT_FILENAME']='SOURCE_DATASET.h5'; os.environ['PACKAGING_OVERWRITE_OUTPUTS']='True'; from helpers.packaging.config import load_packaging_config; from helpers.packaging.pipeline import run_packaging_pipeline; config=load_packaging_config(os.environ); start=time.perf_counter(); output=run_packaging_pipeline(config); elapsed=time.perf_counter()-start; print(f'OUTPUT={output}'); print(f'ELAPSED_SECONDS={elapsed:.6f}')"
```

Result:

- baseline single-shard merge: `584.821280s`

### Baseline cProfile

Command used:

```bash
uv run python -c "import os, cProfile, pstats; from helpers.packaging.config import load_packaging_config; from helpers.packaging.pipeline import run_packaging_pipeline; os.environ['PACKAGING_SOURCE_HDF5_PATH']='/workspace/analysis/stage3_benchmark_single_shard'; os.environ['PACKAGING_OUTPUT_DIR']='/workspace/analysis/stage3_profile_single'; os.environ['PACKAGING_OUTPUT_FILENAME']='SOURCE_DATASET.h5'; os.environ['PACKAGING_OVERWRITE_OUTPUTS']='True'; config=load_packaging_config(os.environ); pr=cProfile.Profile(); pr.enable(); run_packaging_pipeline(config); pr.disable(); pstats.Stats(pr).sort_stats('tottime').print_stats(25)"
```

Top hotspots:

- `h5py/_hl/dataset.py:962(__setitem__)`: `416.996s`
- `h5py/_hl/dataset.py:853(__getitem__)`: `158.036s`
- `helpers/packaging/writer.py:299(merge_source_hdf5_shards)`: `583.617s`

Interpretation:

- The baseline was dominated by per-row HDF5 writes.
- Per-row HDF5 reads were also expensive.
- Python-side provenance hashing was visible but much smaller than the row-by-row HDF5 path.

## Changes Implemented

### 1. Batched Stage 3 reads and writes

- Replaced row-at-a-time merge copies with batched HDF5 transfers.
- Kept deterministic ordering by sorting lightweight row metadata first, then copying payloads in batches.
- Replaced row-at-a-time accepted-manifest filtering with batched copies while preserving manifest row order.

### 2. Lighter merge provenance

- Removed per-image and per-mask SHA256 recomputation inside Stage 3 merge.
- Reused each validated shard's `source_signature` instead.
- Final merge `source_signature` now hashes:
  - ordered row metadata
  - contributing shard paths
  - shard row counts
  - shard `source_signature` values

### 3. Configurable Stage 3 HDF5 compression

- Added `PACKAGING_HDF5_COMPRESSION` with allowed values:
  - `none`
  - `gzip`
  - `lzf`
- Default is now `none` for fastest writes.
- `.env_example` was updated accordingly.

### 4. Configurable Stage 3 copy batch size

- Added `PACKAGING_COPY_BATCH_SIZE` with default `256`.
- This allows workload-specific tuning without changing code.
- `.env_example` was updated accordingly.

### 5. Small cleanup

- Fixed stale Stage 7 labels in Stage 3 logging/error text.

### 5. Slice-based reads for sorted source rows

- After the first refactor, Stage 3 was still mostly read-bound.
- A targeted microbenchmark on the real shard showed that HDF5 fancy indexing with large sorted index arrays was much slower than reading contiguous source slices and concatenating them.
- The merge and filtering paths now convert sorted source row indices into contiguous spans and read each span with slicing.

Microbenchmark on the real single-shard input:

- `np.asarray(dataset[sorted_indices], dtype=np.uint8)`: about `19.55s`
- slice-based reads over contiguous spans plus one `np.concatenate(...)`: about `2.43s`

## After-Change Results

### First optimized pass, single-shard benchmark, default compression (`none`)

Command used:

```bash
uv run python -c "import os, time; os.environ['PACKAGING_SOURCE_HDF5_PATH']='/workspace/analysis/stage3_benchmark_single_shard'; os.environ['PACKAGING_OUTPUT_DIR']='/workspace/analysis/stage3_benchmark_optimized_single'; os.environ['PACKAGING_OUTPUT_FILENAME']='SOURCE_DATASET.h5'; os.environ['PACKAGING_OVERWRITE_OUTPUTS']='True'; from helpers.packaging.config import load_packaging_config; from helpers.packaging.pipeline import run_packaging_pipeline; config=load_packaging_config(os.environ); start=time.perf_counter(); output=run_packaging_pipeline(config); elapsed=time.perf_counter()-start; print(f'OUTPUT={output}'); print(f'ELAPSED_SECONDS={elapsed:.6f}')"
```

Result:

- first optimized single-shard merge (`none`): `77.036148s`
- speedup vs baseline: about `7.59x`

### First optimized pass, single-shard benchmark, forced `gzip`

Command used:

```bash
uv run python -c "import os, time; os.environ['PACKAGING_SOURCE_HDF5_PATH']='/workspace/analysis/stage3_benchmark_single_shard'; os.environ['PACKAGING_OUTPUT_DIR']='/workspace/analysis/stage3_benchmark_optimized_single_gzip'; os.environ['PACKAGING_OUTPUT_FILENAME']='SOURCE_DATASET.h5'; os.environ['PACKAGING_OVERWRITE_OUTPUTS']='True'; os.environ['PACKAGING_HDF5_COMPRESSION']='gzip'; from helpers.packaging.config import load_packaging_config; from helpers.packaging.pipeline import run_packaging_pipeline; config=load_packaging_config(os.environ); start=time.perf_counter(); output=run_packaging_pipeline(config); elapsed=time.perf_counter()-start; print(f'OUTPUT={output}'); print(f'ELAPSED_SECONDS={elapsed:.6f}')"
```

Result:

- first optimized single-shard merge (`gzip`): `84.006611s`
- speedup vs baseline: about `6.96x`

Interpretation:

- The biggest win came from batching and removing redundant merge hashing.
- `none` was still faster than `gzip`, but on this benchmark the structural write-path fix mattered much more than compression alone.

### Second optimized pass, single-shard benchmark, default compression (`none`)

Command used:

```bash
uv run python -c "import os, time; os.environ['PACKAGING_SOURCE_HDF5_PATH']='/workspace/analysis/stage3_benchmark_single_shard'; os.environ['PACKAGING_OUTPUT_DIR']='/workspace/analysis/stage3_benchmark_optimized_single_v2'; os.environ['PACKAGING_OUTPUT_FILENAME']='SOURCE_DATASET.h5'; os.environ['PACKAGING_OVERWRITE_OUTPUTS']='True'; from helpers.packaging.config import load_packaging_config; from helpers.packaging.pipeline import run_packaging_pipeline; config=load_packaging_config(os.environ); start=time.perf_counter(); output=run_packaging_pipeline(config); elapsed=time.perf_counter()-start; print(f'OUTPUT={output}'); print(f'ELAPSED_SECONDS={elapsed:.6f}')"
```

Result:

- second optimized single-shard merge (`none`): `20.211368s`
- speedup vs baseline: about `28.94x`
- speedup vs first optimized pass: about `3.81x`

### Second optimized pass, single-shard benchmark, forced `gzip`

Command used:

```bash
uv run python -c "import os, time; os.environ['PACKAGING_SOURCE_HDF5_PATH']='/workspace/analysis/stage3_benchmark_single_shard'; os.environ['PACKAGING_OUTPUT_DIR']='/workspace/analysis/stage3_benchmark_optimized_single_v2_gzip'; os.environ['PACKAGING_OUTPUT_FILENAME']='SOURCE_DATASET.h5'; os.environ['PACKAGING_OVERWRITE_OUTPUTS']='True'; os.environ['PACKAGING_HDF5_COMPRESSION']='gzip'; from helpers.packaging.config import load_packaging_config; from helpers.packaging.pipeline import run_packaging_pipeline; config=load_packaging_config(os.environ); start=time.perf_counter(); output=run_packaging_pipeline(config); elapsed=time.perf_counter()-start; print(f'OUTPUT={output}'); print(f'ELAPSED_SECONDS={elapsed:.6f}')"
```

Result:

- second optimized single-shard merge (`gzip`): `19.770822s`
- speedup vs baseline: about `29.58x`

Interpretation:

- The slice-based read change was another major win.
- In this same-session benchmark, `none` and `gzip` were very close on wall time after the read-path fix. That likely means the remaining cost is dominated by source reads rather than output compression.
- These later timings should be read with the usual warm-cache caveat, but the cProfile results confirm the algorithmic improvement independently.

### Optimized full configured merge

Command used:

```bash
uv run python -c "import os, time; os.environ['PACKAGING_SOURCE_HDF5_PATH']='/mnt/host_f/CHILE_OUTPUT/PATCHES/HDF5_SHARDS'; os.environ['PACKAGING_OUTPUT_DIR']='/workspace/analysis/stage3_benchmark_optimized_full'; os.environ['PACKAGING_OUTPUT_FILENAME']='SOURCE_DATASET.h5'; os.environ['PACKAGING_OVERWRITE_OUTPUTS']='True'; from helpers.packaging.config import load_packaging_config; from helpers.packaging.pipeline import run_packaging_pipeline; config=load_packaging_config(os.environ); start=time.perf_counter(); output=run_packaging_pipeline(config); elapsed=time.perf_counter()-start; print(f'OUTPUT={output}'); print(f'ELAPSED_SECONDS={elapsed:.6f}')"
```

Result:

- optimized full configured merge: `590.986416s`
- output rows: `7176`

Related baseline observations gathered before the refactor:

- full configured baseline merge did not finish within `1200s`
- a 3-large-shard baseline subset also did not finish within `1800s`

This means the optimized full configured merge now completes in under `10` minutes where the previous path was still not done after `20` to `30` minutes.

### Second optimized pass, full configured merge

Command used:

```bash
uv run python -c "import os, time; os.environ['PACKAGING_SOURCE_HDF5_PATH']='/mnt/host_f/CHILE_OUTPUT/PATCHES/HDF5_SHARDS'; os.environ['PACKAGING_OUTPUT_DIR']='/workspace/analysis/stage3_benchmark_optimized_full_v2'; os.environ['PACKAGING_OUTPUT_FILENAME']='SOURCE_DATASET.h5'; os.environ['PACKAGING_OVERWRITE_OUTPUTS']='True'; from helpers.packaging.config import load_packaging_config; from helpers.packaging.pipeline import run_packaging_pipeline; config=load_packaging_config(os.environ); start=time.perf_counter(); output=run_packaging_pipeline(config); elapsed=time.perf_counter()-start; print(f'OUTPUT={output}'); print(f'ELAPSED_SECONDS={elapsed:.6f}')"
```

Result:

- second optimized full configured merge: `166.181674s`
- output rows: `7176`
- speedup vs first optimized full merge: about `3.56x`

This means the current Stage 3 merge now completes in under `3` minutes on the configured full shard set.

## Output Size Tradeoff

Measured output sizes:

- baseline single-shard output (`gzip`): `104M`
- optimized single-shard output (`none`): `153M`
- optimized single-shard output (`gzip`): `96M`
- optimized full output (`none`): `1.4G`

The second optimized pass preserved the same output-size tradeoff:

- second optimized single-shard output (`none`): `153M`
- second optimized single-shard output (`gzip`): `96M`
- second optimized full output (`none`): `1.4G`

Implication:

- `PACKAGING_HDF5_COMPRESSION=none` is the fastest option.
- It increases file size.
- If storage becomes a concern, `gzip` remains much faster than the old implementation because batching removed the major bottleneck.

## Optimized cProfile

Top hotspots after the refactor on the same single-shard benchmark:

- `h5py/_hl/dataset.py:853(__getitem__)`: `73.877s`
- `h5py/_hl/dataset.py:962(__setitem__)`: `2.416s`
- `helpers/packaging/writer.py:444(merge_source_hdf5_shards)`: `77.197s`

Interpretation:

- Per-row HDF5 writes were effectively eliminated as the dominant bottleneck.
- Stage 3 is now mostly read-bound on the source shard path for this benchmark.

## Second-Pass cProfile

Top hotspots after the slice-based read optimization on the same single-shard benchmark:

- `h5py/_hl/dataset.py:853(__getitem__)`: `13.063s`
- `h5py/_hl/dataset.py:962(__setitem__)`: `3.734s`
- `helpers/packaging/writer.py:489(merge_source_hdf5_shards)`: `18.165s`
- `helpers/packaging/writer.py:85(_read_sorted_rows)`: `13.590s` cumulative

Interpretation:

- Stage 3 is still source-read heavy, but that cost is now far lower than after the first refactor.
- The current implementation is no longer catastrophically dominated by either per-row writes or slow point-selection reads.

## Validation

The following passed after the refactor:

- `uv run pytest tests/test_packaging_config.py tests/test_packaging_writer.py tests/test_packaging_pipeline.py`
- `uv run ruff check helpers/packaging/config.py helpers/packaging/pipeline.py helpers/packaging/writer.py tests/test_packaging_config.py tests/test_packaging_writer.py tests/test_packaging_pipeline.py`
- `uv run mypy helpers/packaging/config.py helpers/packaging/pipeline.py helpers/packaging/writer.py tests/test_packaging_config.py tests/test_packaging_writer.py tests/test_packaging_pipeline.py`

Additional validation after exposing `PACKAGING_COPY_BATCH_SIZE`:

- `uv run pytest tests/test_packaging_config.py tests/test_packaging_writer.py tests/test_packaging_pipeline.py`
- `uv run ruff check helpers/packaging/config.py helpers/packaging/pipeline.py helpers/packaging/writer.py tests/test_packaging_config.py tests/test_packaging_writer.py tests/test_packaging_pipeline.py`
- `uv run mypy helpers/packaging/config.py helpers/packaging/pipeline.py helpers/packaging/writer.py tests/test_packaging_config.py tests/test_packaging_writer.py tests/test_packaging_pipeline.py`

## Batch-Size Tuning

### Full configured merge, `PACKAGING_HDF5_COMPRESSION=none`

Measured with the optimized code by overriding `PACKAGING_COPY_BATCH_SIZE`:

- `64`: `249.314564s`
- `128`: `189.939466s`
- `256`: `153.879494s`
- `512`: `225.258302s`
- `1024`: `154.817833s`

Interpretation:

- `256` and `1024` are effectively tied on the full real workload.
- `512` was worse.
- Returns flatten sharply once the batch size reaches `256`.

### Single-shard merge, `PACKAGING_HDF5_COMPRESSION=none`

- `64`: `37.900806s`
- `128`: `30.359094s`
- `256`: `15.401775s`
- `512`: `15.799996s`
- `1024`: `9.251991s`

Interpretation:

- Larger batches can help on smaller isolated inputs.
- The full workload does not show the same clear win beyond `256`.
- This is why the code keeps a conservative default and exposes batch size as a tuning knob.

## Compression Tuning

### Full configured merge at batch size `256`

- `none`: `230.685612s`
- `lzf`: `267.738781s`
- `gzip`: `383.589624s`

### Single-shard merge at batch size `256`

- `none`: `17.452135s`
- `lzf`: `26.585029s`
- `gzip`: `39.907743s`

Interpretation:

- `none` is the best-performing compression mode.
- `lzf` is consistently slower than `none` and still notably faster than `gzip`.
- `gzip` is the slowest option by a large margin.

## Filtering Path Benchmark

To verify the accepted-manifest path is not a hidden bottleneck, a full accepted-manifest benchmark was run against `/workspace/analysis/stage3_benchmark_optimized_full_v2/SOURCE_DATASET.h5` with a manifest containing all `7176` rows.

### Filtering with `PACKAGING_HDF5_COMPRESSION=none`

- full accepted-manifest filtering: `75.971513s`

### Filtering batch-size sweep

- `256`: `84.509754s`
- `512`: `81.602442s`
- `1024`: `77.811039s`

### Filtering compression sweep at batch size `256`

- `none`: `83.748134s`
- `lzf`: `208.809627s`
- `gzip`: `244.762978s`

Interpretation:

- The filtering path is materially faster than the merge path.
- `none` is also the best compression mode there.
- Larger batches help somewhat, but the gain is modest compared with the already-achieved merge improvements.

## Plateau Assessment

The major Stage 3 performance gains have now been captured.

Evidence for plateau:

- Structural fixes produced very large wins:
  - row-by-row -> batched writes
  - expensive fancy-index reads -> slice-based reads
  - redundant per-patch hashing removed from merge provenance
- Post-optimization tuning results are now much smaller and more workload-dependent:
  - full merge batch-size tuning flattened around `256` to `1024`
  - `none` compression is clearly best and already selected by default
  - filtering-path gains from extra tuning are incremental, not transformative

Practical conclusion:

- The current implementation appears to be at the minimum-return plateau for low-risk code changes.
- Further gains are likely to require deeper HDF5-layout-specific work, more invasive pipeline changes, or a different artifact format, rather than another small Python refactor.

## Recommendation

- Keep `PACKAGING_HDF5_COMPRESSION=none` for the fastest Stage 3 merge and filtering runs.
- Keep `PACKAGING_COPY_BATCH_SIZE=256` as the default.
- If tuning for a specific machine or dataset, try `PACKAGING_COPY_BATCH_SIZE=1024` and compare against `256`.
- Use `gzip` only when the larger output size is a practical issue.
- The biggest confirmed wins for Stage 3 were:
  - batched HDF5 copies instead of row-at-a-time writes
  - removing per-image/per-mask hashing from merge provenance
  - replacing sorted fancy-index reads with slice-based contiguous reads
- Based on the measured batch-size and compression sweeps, Stage 3 has reached the minimum-return plateau for low-risk optimization work.
