from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from helpers.sanity.config import SanityConfig
from helpers.sanity.contracts import (
    check_filename_patient_id_consistency,
    check_filename_uniqueness,
    check_source_reference_contract,
    check_stage5_singleton_layout_contract,
)
from helpers.sanity.disk_checks import (
    check_decode_and_shapes,
    check_manifest_disk_parity,
    check_mask_pixel_values,
    check_paths_exist_and_relative,
    collect_hdf5_row_inspections,
)
from helpers.sanity.manifest_checks import (
    check_duplicate_rows,
    check_manifest_schema,
    check_patient_leakage,
    check_split_constraints_from_run_config,
    check_split_patient_lists_against_run_config,
    check_split_stats_against_manifest,
    check_stage4_cleaning_lineage,
)
from helpers.sanity.models import SPLITS, CheckResult, SanityReport, worst_status
from helpers.sanity.provenance import load_manifest, load_run_config, load_split_stats
from helpers.sanity.semantic_checks import (
    check_class_balance_visibility,
    check_mask_label_semantics,
    check_patches_per_patient_stats,
    check_patient_level_balance,
    check_split_class_presence,
)


def _split_dataframe(manifest_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return cast(pd.DataFrame, manifest_df.loc[manifest_df["split"] == split_name].copy())


def run_sanity_pipeline(config: SanityConfig) -> SanityReport:
    base_dir = Path(config.base_dir)
    manifest_df = load_manifest(base_dir)
    run_config = load_run_config(base_dir)
    split_stats_df = load_split_stats(base_dir)
    dataset_checks = {
        "Manifest Schema": check_manifest_schema(manifest_df),
        "Duplicate Rows": check_duplicate_rows(manifest_df),
        "Patient Leakage": check_patient_leakage(manifest_df),
        "Stage 5 Singleton Layout": check_stage5_singleton_layout_contract(manifest_df),
        "RunConfig Patient Lists": check_split_patient_lists_against_run_config(
            manifest_df, run_config
        ),
        "RunConfig Constraints": check_split_constraints_from_run_config(manifest_df, run_config),
    }
    if config.enforce_filename_uniqueness:
        dataset_checks["Filename Uniqueness"] = check_filename_uniqueness(manifest_df)
    if config.enforce_regex_patient_id_match:
        dataset_checks["Filename Patient-ID Contract"] = check_filename_patient_id_consistency(
            manifest_df
        )
    if config.enforce_split_stats_parity:
        dataset_checks["Split Stats Parity"] = check_split_stats_against_manifest(
            manifest_df, split_stats_df
        )
    dataset_checks["Source Reference Contract"] = check_source_reference_contract(manifest_df)
    dataset_checks["Stage4 Cleaning Lineage"] = check_stage4_cleaning_lineage(
        manifest_df,
        run_config,
        base_dir,
    )

    split_checks: dict[str, dict[str, CheckResult]] = {}
    for split_name in SPLITS:
        split_df = _split_dataframe(manifest_df, split_name)
        row_inspections = collect_hdf5_row_inspections(split_df, base_dir, split_name)
        split_checks[split_name] = {
            "Disk Parity": check_manifest_disk_parity(split_df, base_dir, split_name),
            "Paths Exist": check_paths_exist_and_relative(
                split_df, base_dir, sample_n=config.sample_pairs
            ),
            "Decode & Shapes": check_decode_and_shapes(
                split_df,
                base_dir,
                split_name,
                sample_n=config.sample_pairs,
                full_scan=config.full_shape_scan,
                row_inspections=row_inspections,
            ),
            "Mask Pixel Values": check_mask_pixel_values(
                split_df,
                base_dir,
                split_name,
                sample_n=config.sample_pairs,
                full_scan=config.full_mask_scan,
                row_inspections=row_inspections,
            ),
            "Mask/Label Semantics": check_mask_label_semantics(
                split_df,
                base_dir,
                split_name,
                fail_on_empty_cancer_mask=config.fail_on_empty_cancer_mask,
                fail_on_positive_not_cancer_mask=config.fail_on_positive_not_cancer_mask,
                row_inspections=row_inspections,
            ),
            "Patch Class Balance": check_class_balance_visibility(split_df),
            "Patient Class Balance": check_patient_level_balance(split_df),
            "Split Class Presence": check_split_class_presence(split_df),
            "Patches/Patient": check_patches_per_patient_stats(split_df),
        }

    all_statuses = [result.status for result in dataset_checks.values()]
    for split_name in SPLITS:
        all_statuses.extend(result.status for result in split_checks[split_name].values())
    verdict = "SPLITS PASSED" if worst_status(all_statuses) != "FAIL" else "SPLITS REJECTED"
    return SanityReport(
        base_dir=str(base_dir),
        dataset_checks=dataset_checks,
        split_checks=split_checks,
        verdict=verdict,
    )
