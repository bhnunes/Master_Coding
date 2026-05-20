# Method Sources

This file maps the current Elsevier manuscript in `reports/main.tex` to the
thesis chapters, final result artifacts, and repository modules that support its
main claims. It is intentionally concise: the paper should remain cleaner than
the thesis while retaining enough traceability to audit every quantitative and
methodological statement.

## Manuscript Narrative

- `masters_thesis/New_Thesis/resumo.tex`
- `masters_thesis/New_Thesis/desenvolvimento.tex`
- `masters_thesis/New_Thesis/resultados.tex`
- `masters_thesis/New_Thesis/conclusoes.tex`
- `references/citations.md`

The paper-level claim is methodological: an auditable end-to-end workflow for
digital pathology experiments, evaluated on DIAGSET. It should not claim clinical
deployment readiness or superiority over external baselines.

## Dataset, Curation, and Split Evidence

- `masters_thesis/New_Thesis/resultados.tex`
  - ingestion: 429 images, 429 annotations, 419 valid GeoJSON-paired cases
  - graph-review samples: pilot 94/6, master 367/18 approved/rejected
  - graph-cleaning parameters: background threshold 167, `k=140`, minimum size 97,
    erosion 3 px, contamination threshold 0.19
  - retained dataset: 1,186,778 accepted patches after 74,047 cancer-patch rejections
  - patient split: TRAIN/VALIDATION/TEST = 236/78/79 patients
  - test set: 38 positive and 41 negative patients, 387,325 patches
  - smart sampling: 180,381 retained of 660,867 training patches
- `helpers/extraction/*`
- `helpers/graph/*`
- `helpers/crossfold/*`
- `helpers/smart_sampling/*`

## Training, Ensemble, and Metrics Evidence

- `masters_thesis/New_Thesis/resultados.tex`
  - five normalization settings: no normalization, Reinhard, Ruifrok, Macenko,
    Vahadane
  - 40 final trainings, eight architectures per normalization, seed 24
  - ensemble composition and thresholds for each normalization
  - final micro, macro, AUROC, positive-patient Dice, and negative-clean-rate
    findings
- Final metric JSON files under
  `D:/Usuario/Desktop/Results_DIAGSET/DIAGSET_RESULTS/DIAGSET_RESULTS/*/INFERENCE/`
  - `ENSEMBLE_METRICS_11_05_2026_14_51_19.json` for `NOT_NORMALIZED`
  - `ENSEMBLE_METRICS_11_05_2026_15_16_10.json` for `REINHARD`
  - `ENSEMBLE_METRICS_12_05_2026_13_11_34.json` for `RUIFROK`
  - `ENSEMBLE_METRICS_12_05_2026_13_13_22.json` for `MACENKO`
  - `ENSEMBLE_METRICS_12_05_2026_13_17_25.json` for `VAHADANE`
- `helpers/training/*`
- `helpers/ensemble_optimizer/*`
- `helpers/ensemble_inference/*`

The compact final-results table in `reports/main.tex` was cross-checked against
the thesis tables and the `ENSEMBLE_METRICS_*.json` files listed above.

## Paper Figures

Paper-selected assets are copied or rendered into `reports/figures/`:

- `pipeline_overview.png` from `MERMAID/pipeline_overview.mermaid`
- `results_diagset_micro_metrics.png` from `masters_thesis/New_Thesis/figuras/`
- `results_diagset_macro_metrics.png` from `masters_thesis/New_Thesis/figuras/`

The paper intentionally omits LR-finder plots, per-architecture checkpoint
tables, and the full thesis confusion-matrix set to keep the Elsevier manuscript
focused.

## Citation Sources

- `reports/references.bib`
- `masters_thesis/New_Thesis/bibliografia.bib`
- `references/citations.md`

The report-local bibliography uses vetted peer-reviewed or proceedings entries
needed by the paper narrative. ArXiv-only entries are excluded by default.

## Build Contract

Build from `reports/` with the Elsevier template visible:

```powershell
$env:TEXINPUTS = '../latex_template//;'
$env:BSTINPUTS = '../latex_template//;'
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

After building, inspect `main.log` for undefined citations, undefined
references, missing figures, LaTeX errors, and serious overfull boxes.
