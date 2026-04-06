# Stage 7 Scientific Notes

## Scientific Interpretation

Stage 7 is a patient-wise, label-aware training-set curation step.

It should not be described as a purely unsupervised diversity sampler.
The current Stage 7 behavior intentionally preserves clinically important rows
before reducible sampling by allowing two protection policies:

- positive-label retention
- mask-positive retention above `SMART_SAMPLER_POSITIVE_MASK_FRACTION_THRESHOLD`

This means downstream model changes after enabling Stage 7 may reflect a
combination of:

- improved embedding-space coverage of reducible rows
- changed retained class composition
- changed patient-level patch weighting

Any report, benchmark, or manuscript using Stage 7 should state that the
training set was filtered with a label-aware policy.

## Large-Patient Selection Behavior

For larger patients, the stability loop is used to estimate whether the
embedding representation has stabilized, but the final selector now operates on
the full reducible patient pool rather than a single accepted random subset.

This avoids the earlier scientific risk where rare morphologies could be
excluded from the final candidate set simply because they were absent from an
intermediate random subset.

## Holdout Terminology

Adaptive keep stopping uses a within-patient patch holdout.

This is not an independent patient-level validation split and should not be
described as such. The exported provenance uses the term:

- `within_patient_patch_holdout`

## Provenance Written by Stage 7

Stage 7 now writes durable methodology metadata directly into
`TRAIN_FILTERED.h5` as HDF5 attributes. These fields are intended to survive
even if sidecar CSV/JSON files are lost.

Current Stage 7 HDF5 provenance includes:

- `stage7_label_aware`
- `stage7_selector`
- `stage7_model_name`
- `stage7_seed`
- `stage7_stability_threshold`
- `stage7_stability_repeats`
- `stage7_keep_improvement_threshold`
- `stage7_keep_patience`
- `stage7_keep_min`
- `stage7_keep_step`
- `stage7_m_max`
- `stage7_holdout_mode`
- `stage7_protect_positive_labels`
- `stage7_protect_mask_positive`
- `stage7_positive_mask_fraction_threshold`

The `selection_signature` also incorporates Stage 7 methodology metadata so
filtered-HDF5 reuse remains fail-closed when Stage 7 settings change.

## Sidecar Fields Worth Reporting

When analyzing Stage 7 outputs, the most useful sidecar fields are:

- `protected_count`
- `protected_positive_label_count`
- `protected_mask_positive_count`
- `reducible_count`
- `selected_positive_label_count`
- `selected_negative_label_count`
- `adaptive_m_target`
- `heldout_count`
- `plateau_threshold`
- `plateau_trigger_improvement`
- `plateau_trigger_keep_count`
- `plateau_trigger_step`
- `plateau_stop_reason`
- `plateau_evaluation_mode`

At the run level, `filter_summary.json` now captures:

- protected retained totals
- selected positive/negative totals
- retained positive/negative fractions
- the Stage 7 label-aware flag
- the holdout evaluation mode

## Reporting Guidance

Recommended wording for methods sections:

"Stage 7 applied patient-wise, label-aware smart sampling to the training HDF5.
Positive-label and mask-positive patches were protected from reduction, while
the remaining reducible patches were filtered with embedding-based selection.
Adaptive stopping used a within-patient patch holdout and the final selector
operated on the full reducible patient pool."
