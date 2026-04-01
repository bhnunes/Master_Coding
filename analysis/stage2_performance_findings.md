# Stage 2 Performance Findings

## Scope

- Focus only on algorithmic/code speedups for `2_database_manager.py` hot path.
- Keep the existing `.env` paths unchanged during this analysis.
- Benchmark only these database IDs: `8, 4, 6, 7, 2, 5, 1`.
- Ignore small optimizations; record only bottlenecks or changes with potential for major wins.

## Current Benchmark Protocol

- Runner: `uv run python scripts/benchmark_stage2_cases.py`
- Environment: current `.env`
- Output directory: `analysis/stage2_benchmarks/`
- Per-slide extraction profiles: `analysis/stage2_benchmarks/profiles/`
- Agreed benchmark set: `8, 4, 6, 7, 2, 5, 1`
- Useful switches:
  - `--skip-hdf5-write` to isolate extraction from post-extraction HDF5 time
  - `STAGE2_HDF5_COMPRESSION=gzip|lzf|none`
  - `OPENSLIDE_CACHE_BYTES=<bytes>`
  - `STAGE2_PRELOAD_SCAN_AREA_MAX_BYTES=<bytes>`

## Confirmed Observations

- `2_database_manager.py` itself is mostly orchestration.
- The Stage 2 hot path is `helpers/extraction/patch_engine.py`, reached through `run_slide_processing()`.
- Existing profiling support already captures slide-level and per-window phase timing, so initial investigation can be evidence-driven.
- The first full baseline showed a critical split between extraction time and end-to-end wall time.
- For the two slowest requested slides before optimization:
  - case `1`: wall time `631.213s`, extraction profile `12.948s`
  - case `5`: wall time `381.751s`, extraction profile `6.825s`
- That means the original dominant bottleneck was not patch extraction itself. It was the post-extraction HDF5 shard writing path.

## Baseline Evidence

- Baseline benchmark summary: `analysis/stage2_benchmarks_baseline/summary.json`
- Extraction-only control for case `1` with HDF5 disabled:
  - `uv run python scripts/benchmark_stage2_cases.py --case-ids 1 --skip-hdf5-write ...`
  - result: wall time `13.438s`, extraction profile `13.353s`
- This confirmed that nearly all of the original `631s` runtime for case `1` was outside extraction and inside the HDF5 writing stage.

## Root Cause Found

- `helpers/extraction/hdf5_storage.py` originally created the HDF5 datasets once and then wrote rows one by one:
  - one write per image row
  - one write per mask row
  - one write per metadata row and dataset cell
- That produced thousands of Python-to-HDF5 write calls per slide.
- On real Stage 2 workloads this was catastrophic for end-to-end runtime.

## Major Improvement Implemented

- Refactored `write_slide_patch_dataset_hdf5()` to batch materialize arrays and write each dataset in one operation instead of row by row.
- Preserved:
  - patch counts
  - image and mask content
  - canonical dataset names
  - source signature generation
  - manifest behavior

## Before / After Results

- Full post-change benchmark summary: `analysis/stage2_benchmarks_after_batch_write/summary.json`

- Case `1`
  - before: `631.213s`
  - after: `20.397s`
  - speedup: about `30.95x`
  - patch counts unchanged: `91` cancer, `695` not-cancer

- Case `5`
  - before: `381.751s`
  - after: `14.251s`
  - speedup: about `26.79x`
  - patch counts unchanged: `69` cancer, `500` not-cancer

- Full requested benchmark set
  - before total: `1079.635s`
  - after total: `92.519s`
  - speedup: about `11.67x`

## Second Major Improvement

- Added configurable Stage 2 HDF5 compression via `STAGE2_HDF5_COMPRESSION`.
- Benchmarked on case `1` sequentially:
  - `gzip`: `22.256s`
  - `lzf`: `20.142s`
  - `none`: `14.469s`
- Patch counts remained unchanged across compression modes.
- Based on that result, the active `.env` now sets:
  - `STAGE2_HDF5_COMPRESSION=none`

## Full Benchmark With No HDF5 Compression

- Summary: `analysis/stage2_benchmarks_full_none/summary.json`
- Full requested benchmark set with `STAGE2_HDF5_COMPRESSION=none`:
  - total: `78.397s`
  - relative to original baseline `1079.635s`: about `13.77x` faster

- Notable slides:
  - case `1`: `14.494s` vs original `631.213s` (`43.55x` faster)
  - case `5`: `10.735s` vs original `381.751s` (`35.56x` faster)

## Current Best Known Configuration

- Active `.env` choice for fastest proven Stage 2 writes:
  - `STAGE2_HDF5_COMPRESSION=none`
- Keep these disabled unless specifically benchmarking them:
  - `OPENSLIDE_CACHE_BYTES=0`
  - `STAGE2_PRELOAD_SCAN_AREA_MAX_BYTES=0`
- Representative benchmark artifact for the current best-known path:
  - `analysis/stage2_benchmarks_full_none/summary.json`

## Current Post-Optimization Bottlenecks

- After the writer refactor, the remaining dominant extraction phase is `parallel_processing` inside `patch_engine.py`.
- Within extraction, `read_region` is now the largest timed window phase on representative slides.
- Remaining likely high-impact areas, if further work is needed:
  - reduce unnecessary `read_region` calls
  - reduce accepted-patch IPC payload size
  - reduce patch-engine work for windows that are later rejected

## Open Questions

- How much of the remaining `parallel_processing` time is OpenSlide reads versus worker payload transfer?
- Can extraction avoid some full patch reads while preserving exact saved patch content?

## Negative Results / Constraints

- OpenSlide cache support was wired in as an optional path via `OPENSLIDE_CACHE_BYTES`, but the current runtime uses an older native OpenSlide build that does not support `OpenSlideCache`.
- The cache path therefore remains disabled by default (`OPENSLIDE_CACHE_BYTES=0`).
- Worker-count sweep on case `1` with HDF5 disabled did not show meaningful gains:
  - `NUM_WORKERS=1`: `9.356s`
  - `NUM_WORKERS=3`: `9.295s`
  - `NUM_WORKERS=6`: `9.413s`
  - `NUM_WORKERS=8`: `9.268s`
- That suggests the remaining extraction hotspot is not currently limited by simple worker parallelism.
- Preloading the full optimized scan area into memory was tested via `STAGE2_PRELOAD_SCAN_AREA_MAX_BYTES`, but it was dramatically slower on real slides:
  - case `1`: about `71.5s`
  - case `2`: about `176.8s`
- The preload path remains available but disabled by default (`STAGE2_PRELOAD_SCAN_AREA_MAX_BYTES=0`).
- A true in-process serial extraction path for `NUM_WORKERS=1` was also tested and did not outperform the previous pool-based path materially on case `1`.
- A streaming `source_signature` rewrite was tested and reverted because it did not beat the current writer on the real slow slides.

## Docker / OpenSlide Status

- Before the Docker update, the runtime reported:
  - `openslide-python`: `1.4.3`
  - native OpenSlide: `3.4.1`
- That native version was too old for `OpenSlideCache`, which requires native OpenSlide `>= 4.0.0`.
- `Dockerfile` has now been updated to build native OpenSlide `4.0.0` from source and to fail the image build unless:
  - native OpenSlide is `>= 4.0.0`
  - `openslide.OpenSlideCache(...)` can be instantiated
- This Docker change has now been validated in a rebuilt container:
  - `openslide-python`: `1.4.3`
  - native OpenSlide: `4.0.0`
  - `openslide.OpenSlideCache(...)` instantiates successfully

## OpenSlide Cache Benchmark After Rebuild

- Extraction-only benchmark command remained:
  - `uv run python scripts/benchmark_stage2_cases.py --case-ids 1 --skip-hdf5-write ...`
- Fixed settings during this cache sweep:
  - `STAGE2_HDF5_COMPRESSION=none`
  - `STAGE2_PRELOAD_SCAN_AREA_MAX_BYTES=0`
  - current `.env` paths unchanged
- Case `1` patch counts stayed unchanged for every cache size tested:
  - `91` cancer
  - `695` not-cancer

- First rebuilt-container sweep on case `1`:
  - `OPENSLIDE_CACHE_BYTES=0`: extraction `19.047s`
  - `OPENSLIDE_CACHE_BYTES=134217728`: extraction `20.604s`
  - `OPENSLIDE_CACHE_BYTES=536870912`: extraction `20.796s`
- Follow-up same-session confirmation rerun on case `1`:
  - `OPENSLIDE_CACHE_BYTES=0`: extraction `13.849s`
  - `OPENSLIDE_CACHE_BYTES=134217728`: extraction `14.151s`

- Profile comparison from the confirmation rerun:
  - `OPENSLIDE_CACHE_BYTES=0`
    - `parallel_processing`: `12.284s`
    - aggregated `read_region`: `25.350s`
  - `OPENSLIDE_CACHE_BYTES=134217728`
    - `parallel_processing`: `12.609s`
    - aggregated `read_region`: `26.284s`
  - `OPENSLIDE_CACHE_BYTES=536870912` (first sweep)
    - `parallel_processing`: `19.276s`
    - aggregated `read_region`: `39.358s`

- Conclusion:
  - OpenSlide cache did not improve Stage 2 extraction on the rebuilt container for the representative slow slide.
  - `OPENSLIDE_CACHE_BYTES=0` remains the best-known setting.
  - `134217728` was consistently slightly slower.
  - `536870912` was materially slower.
  - Do not enable cache by default based on current evidence.

## Best Next Step

- Keep `OPENSLIDE_CACHE_BYTES=0` as the default.
- Continue with code-level extraction work inside `helpers/extraction/patch_engine.py`.
- Most likely next areas to validate are still:
  - reduce worker-to-parent transfer of accepted image and mask arrays if IPC remains material
  - separate overlap decision from full patch materialization so rejected windows avoid more work
  - strengthen early geometric filtering before expensive mask clipping / rasterization

## Candidate Major Improvements To Validate

- Reduce worker-to-parent transfer of accepted image and mask arrays if profiling shows IPC overhead is still material.
- Separate overlap decision from full mask materialization so rejected windows do less work.
- Strengthen early geometric filtering before expensive clipping / rasterization.

## Third Confirmed Improvement

- In `helpers/extraction/patch_engine.py`, the hot `read_region(...).convert("RGB")` path originally converted the PIL image to NumPy with `np.array(...)`.
- Replaced that conversion with `np.asarray(...)` for both per-patch reads and the optional preloaded scan-area read.
- A direct microbenchmark on a real case `1` slide showed the conversion step itself was much cheaper with identical pixel output for the sampled patch:
  - `np.array(slide.read_region(...).convert("RGB"))`: about `0.028s` per call in the microbenchmark
  - `np.asarray(slide.read_region(...).convert("RGB"))`: about `0.003s` per call in the microbenchmark
- Spot checks on the real slide confirmed identical RGB arrays for the tested patch.

## Full Benchmark After `np.asarray` Conversion

- Summary: `analysis/stage2_benchmarks_full_none_asarray/summary.json`
- Full requested benchmark set with:
  - `STAGE2_HDF5_COMPRESSION=none`
  - `OPENSLIDE_CACHE_BYTES=0`
  - `STAGE2_PRELOAD_SCAN_AREA_MAX_BYTES=0`
- Total wall time:
  - before: `78.397569s`
  - after: `73.357255s`
  - speedup: about `1.07x` faster (`6.4%` lower wall time)
- Patch counts remained unchanged for every benchmarked slide.

- Per-slide end-to-end results:
  - case `8`: `3.905s` -> `3.565s`
  - case `4`: `6.458s` -> `5.253s`
  - case `6`: `9.465s` -> `8.776s`
  - case `7`: `10.287s` -> `9.738s`
  - case `2`: `23.053s` -> `21.831s`
  - case `5`: `10.735s` -> `10.037s`
  - case `1`: `14.494s` -> `14.158s`

- Representative extraction profile change for case `1`:
  - previous extraction: `8.965s`
  - new extraction: `8.572s`
  - `parallel_processing`: `7.933s` -> `7.671s`
  - aggregated `read_region`: `16.380s` -> `15.959s`
