# No-Rerun Peer-Review Evidence Index

This file records how the revised manuscript addresses likely peer-review
questions using only existing manuscript text, thesis text, code-level protocol
evidence, and frozen DIAGSET result artifacts. No experiments, model training,
inference runs, threshold searches, or bootstrap recomputations were performed
for this revision.

## Reviewer-Response Matrix

| Peer-review concern | Existing evidence source | Manuscript action | Residual limitation |
|---|---|---|---|
| Claims may be too broad for one dataset. | `reports/main.tex`; `reports/method_sources.md`; DIAGSET result JSONs. | Title, abstract, discussion, and conclusion now frame the work as a DIAGSET prostate segmentation evaluation. | No external dataset validation is available. |
| Graph-cleaning curation may be viewed as leakage-prone if tuned before final splitting. | `reports/main.tex`; `masters_thesis/New_Thesis/resultados.tex`; `helpers/graph/tuning_pipeline.py`. | Methods now treat graph cleaning as pre-split annotation/background curation and explicitly avoid claiming train-only tuning. | The DIAGSET result bundle does not preserve a complete reviewed-stem-to-final-split audit. |
| Validation set may be overused. | `ENSEMBLE_TWO_STREAM_*.json`; `ensemble_optimizer_run_config.json`. | Methods now include a validation-burden table: 78 validation patients, 20 calibration patients, 50 semantic and 50 spatial trials per normalization, and frozen target statuses. | No nested validation or new held-out calibration split can be added without rerunning experiments. |
| Metric interpretation could hide empty-mask behavior. | `ENSEMBLE_METRICS_*.json`; `helpers/ensemble_inference/metrics.py`; thesis results text. | Metrics section now distinguishes post-processed Dice/NCR from raw-probability AUROC and defines negative-patient Dice handling. | AUROC remains a point estimate because patient-indexed probability distributions were not preserved. |
| Normalization comparison could imply superiority. | Final metric JSONs and bootstrap CIs. | Winner language and bold table emphasis were removed; text says "largest point estimate" and "descriptive." | No paired patient-level difference intervals or repeated-seed analysis are available. |
| DIAGSET and pathologist contextual comparisons may overreach. | `reports/main.tex`; cited DIAGSET and clinical pathology references. | DIAGSET comparison is kept as protocol-aware context; the numeric pathologist table was removed. | The workflow result is still not commensurate with WSI/slide/case-level clinical readings. |
| Workflow component contributions are unclear. | Existing integrated workflow artifacts only. | Discussion now states that current artifacts do not isolate causal contributions of graph cleaning, smart sampling, two-stream gating, or patient suppression. | No ablation study can be added without rerunning experiments. |
| Reproducibility depends on large artifacts not packaged with the paper. | Local result bundle, model registry JSONs, repository commit. | Data availability now points to this no-rerun evidence index and identifies which frozen artifacts support manuscript tables. | HDF5 data, checkpoints, and inference outputs remain too large for the manuscript source package. |

## Final Metric Artifact Check

The compact final-results table in `reports/main.tex` was checked against the
following frozen inference metrics:

| Normalization | Metric artifact | Micro Dice | Macro Dice | AUROC | Positive-patient Dice | NCR | Patch AvAcc |
|---|---|---:|---:|---:|---:|---:|---:|
| None | `D:/Usuario/Desktop/DIAGSET_RESULTS_UPDATED/DIAGSET_RESULTS/NOT_NORMALIZED/INFERENCE/ENSEMBLE_METRICS_25_05_2026_13_22_56.json` | 0.8415 [0.7767, 0.8987] | 0.7540 [0.6710, 0.8356] | 0.9508 | 0.7781 [0.6816, 0.8628] | 0.7317 [0.5909, 0.8649] | 0.8923 |
| Reinhard | `D:/Usuario/Desktop/DIAGSET_RESULTS_UPDATED/DIAGSET_RESULTS/REINHARD/INFERENCE/ENSEMBLE_METRICS_26_05_2026_12_59_29.json` | 0.8296 [0.7773, 0.8789] | 0.7363 [0.6510, 0.8192] | 0.9445 | 0.7413 [0.6385, 0.8327] | 0.7317 [0.5909, 0.8649] | 0.8811 |
| Ruifrok | `D:/Usuario/Desktop/DIAGSET_RESULTS_UPDATED/DIAGSET_RESULTS/RUIFROK/INFERENCE/ENSEMBLE_METRICS_26_05_2026_14_56_02.json` | 0.8295 [0.7786, 0.8710] | 0.5993 [0.5038, 0.6948] | 0.9579 | 0.7985 [0.7189, 0.8688] | 0.4146 [0.2666, 0.5641] | 0.8997 |
| Macenko | `D:/Usuario/Desktop/DIAGSET_RESULTS_UPDATED/DIAGSET_RESULTS/MACENKO/INFERENCE/ENSEMBLE_METRICS_26_05_2026_16_18_56.json` | 0.8000 [0.7198, 0.8510] | 0.4894 [0.3921, 0.5878] | 0.9534 | 0.7806 [0.6930, 0.8585] | 0.2195 [0.1000, 0.3500] | 0.9009 |
| Vahadane | `D:/Usuario/Desktop/DIAGSET_RESULTS_UPDATED/DIAGSET_RESULTS/VAHADANE/INFERENCE/ENSEMBLE_METRICS_26_05_2026_14_48_03.json` | 0.7862 [0.7167, 0.8433] | 0.4405 [0.3454, 0.5378] | 0.9389 | 0.7579 [0.6716, 0.8362] | 0.1463 [0.0476, 0.2632] | 0.8715 |

AUROC source in these files is `raw_probabilities_before_hard_postprocessing`.
Dice, positive-patient Dice, NCR, and patch metrics are computed after the
frozen threshold/component/patient-suppression recipe described in the paper.

## Optimizer And Calibration Artifact Check

The validation-burden table in `reports/main.tex` was checked against the
following optimizer artifacts:

| Normalization | Optimizer recipe artifact | Run config artifact | Validation patients | Calibration patients | Semantic trials | Spatial trials | Manuscript wording | Exact optimizer code |
|---|---|---|---:|---:|---:|---:|---|---|
| None | `.../NOT_NORMALIZED/OPTIMIZATION/ENSEMBLE_TWO_STREAM_25_05_2026_06_36_40.json` | `.../NOT_NORMALIZED/OPTIMIZATION/ensemble_optimizer_run_config.json` | 78 | 20 | 50 | 50 | Required micro-Dice guardrail could not be satisfied | `micro_guardrail_infeasible` |
| Reinhard | `.../REINHARD/OPTIMIZATION/ENSEMBLE_TWO_STREAM_26_05_2026_11_00_38.json` | `.../REINHARD/OPTIMIZATION/ensemble_optimizer_run_config.json` | 78 | 20 | 50 | 50 | Calibration target met | `target_met` |
| Ruifrok | `.../RUIFROK/OPTIMIZATION/ENSEMBLE_TWO_STREAM_26_05_2026_14_34_48.json` | `.../RUIFROK/OPTIMIZATION/ensemble_optimizer_run_config.json` | 78 | 20 | 50 | 50 | Calibration target met | `target_met` |
| Macenko | `.../MACENKO/OPTIMIZATION/ENSEMBLE_TWO_STREAM_26_05_2026_16_00_11.json` | `.../MACENKO/OPTIMIZATION/ensemble_optimizer_run_config.json` | 78 | 20 | 50 | 50 | Required micro-Dice guardrail could not be satisfied | `micro_guardrail_infeasible` |
| Vahadane | `.../VAHADANE/OPTIMIZATION/ENSEMBLE_TWO_STREAM_26_05_2026_14_25_27.json` | `.../VAHADANE/OPTIMIZATION/ensemble_optimizer_run_config.json` | 78 | 20 | 50 | 50 | Calibration target could not be satisfied | `target_infeasible` |

## Curation Provenance Search

Manual review evidence available to the manuscript includes reviewed sample
counts, selected graph parameters, and thesis/source-text descriptions:

- pilot review: 100 samples, 94 approved and 6 rejected
- master review: 385 samples, 367 approved and 18 rejected
- selected graph parameters: background threshold 167, Felzenszwalb `k=140`,
  minimum segment size 97, erosion 3 px, contamination threshold `tau=0.19`

No `APPROVED`/`REJECTED` review-label folders or `graph_cleaning_params.json`
file were found inside `D:/Usuario/Desktop/DIAGSET_RESULTS_UPDATED`. A separate
legacy `D:/Usuario/Desktop/Repositorio_Mestrado/.../IMAGES/APPROVED` and
`REJECTED` pair exists, but it is not identifiable here as the preserved DIAGSET
graph-cleaning review source and was not used as evidence. Therefore, the
manuscript uses conservative limitation language instead of reconstructing a
reviewed-stem-to-final-split audit by assumption.

## Public Reproducibility Anchors

- Repository commit used for this revision: `082e0a07c6b23e05659b4f46b5e1a62dff20fae9`
- Model registry snapshots:
  - `training_model_registry_NOT_NORMALIZED.json`
  - `training_model_registry_REINHARD.json`
  - `training_model_registry_RUIFROK.json`
  - `training_model_registry_MACENKO.json`
  - `training_model_registry_VAHADANE.json`
- Manuscript source: `reports/main.tex`
- Claim/source map: `reports/method_sources.md`
- Thesis result source used for narrative and table cross-checks:
  `masters_thesis/New_Thesis/resultados.tex`

## Code-Level Protocol Evidence Used Only For Description

- Graph tuning uses grouped internal train/test and cross-validation splits:
  `helpers/graph/tuning_pipeline.py`.
- Final patient-level split and split validation are Stage 4/5 repository
  contracts: `helpers/crossfold/*` and `helpers/sanity/*`.
- Runtime normalization is selected downstream and uses split-bundle state:
  `helpers/training/stain_normalization.py`.
- Smart sampling is a train-only downstream selection contract:
  `helpers/smart_sampling/*`.
- Frozen ensemble inference and reporting are implemented in
  `helpers/ensemble_optimizer/*` and `helpers/ensemble_inference/*`.
