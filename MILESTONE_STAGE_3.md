## MILESTONE 1 - Switch Stage 3.1 To Master Manifest Discovery --DONE

- Replace Stage 3.1 shard-scanning discovery with `master_manifest.sqlite` queries.
- Remove the Stage 3.1 assumption that raw `HDF5_SHARDS/` is the direct source of truth.
- Make Stage 3.1 read canonical row identity from the manifest: `filename`, `label`, `source_hdf5_path`, `source_row_index`, `source_image_path`, and `source_mask_path`.

## MILESTONE 2 - Restrict Stage 3.1 To Cancer Patches Only --DONE

- Filter the Stage 3.1 candidate population to `label = 1` only.
- Ensure the Stage 3.1 reported population statistics and sample-size calculations are computed from the cancer-only population.
- Ensure generated pilot and master-pool overlays correspond only to cancer rows.

## MILESTONE 3 - Update Stage 3.1 Config, Docs, And Tests --DONE

- Replace the Stage 3.1 input contract so configuration points to `master_manifest.sqlite` instead of `HDF5_SHARDS/`.
- Remove obsolete Stage 3.1 shards-only config and tests.
- Update `.env_example` and Stage 3.1 tests to reflect manifest-driven cancer-only sampling.

## MILESTONE 4 - Switch Stage 3.2 To Manifest-Aligned Resolution --DONE

- Keep Stage 3.2 review-label resolution aligned with the Stage 3.1 manifest-driven cancer-only population.
- Remove stale Stage 3.2 assumptions that the candidate population should be discovered directly from raw shard enumeration.
- Ensure Stage 3.2 tunes only from reviewed cancer-patch overlays produced by Stage 3.1.

## MILESTONE 5 - Switch Stage 3.3 To Master Manifest Discovery --DONE

- Replace Stage 3.3 shard-scanning candidate discovery with `master_manifest.sqlite` queries.
- Use canonical manifest row identity to load the corresponding HDF5 image and mask refs for scoring.
- Preserve efficient batched HDF5 scoring where possible while using manifest rows as the source of truth.

## MILESTONE 6 - Restrict Stage 3.3 To Cancer Patches Only --DONE

- Apply Stage 3.3 graph cleaning only to manifest rows with `label = 1`.
- Do not score or update non-cancer rows as part of this cancer-cleaning workflow.
- Ensure Stage 3.3 summary counts reflect the cancer-only evaluated population.

## MILESTONE 7 - Keep Manifest Updates As The Canonical Output --DONE

- Preserve the current Stage 3.3 behavior of updating `patch_stage_state` in `master_manifest.sqlite`.
- Do not rewrite or delete rows inside Stage 2 HDF5 shard files.
- Ensure accepted/rejected decisions continue to map back to canonical manifest rows by `source_hdf5_path` and `source_row_index`.

## MILESTONE 8 - Remove Obsolete Raw-Shard Source Contracts --DONE

- Remove Stage 3.x code paths, docs, and tests that treat raw shard enumeration as the source of truth for candidate selection.
- Keep HDF5 shard access only as the pixel-loading backend for rows already selected from the manifest.
- Rename config fields and function parameters where needed so the code clearly reflects the manifest-driven contract.

## MILESTONE 9 - Validate The Full Stage 3 Contract --DONE

- Add or update regression tests proving Stage 3.1 and Stage 3.3 operate on cancer-only manifest rows.
- Add or update regression tests proving Stage 3.3 updates `master_manifest.sqlite` for the correct canonical rows.
- Run targeted `pytest`, `ruff check`, `ruff format`, and `mypy` on all touched Stage 3 files.
