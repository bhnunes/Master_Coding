# Pipeline Citation Register

## Purpose

This note is a working reference for papers that should support the thesis and
`reports/main.tex`. It records what each source should be cited for, where the
local PDF came from when available, and what not to overclaim.

The goal is not to make every pipeline choice look copied from a paper. Several
choices are implementation-specific. Use these citations to explain motivation,
accepted practice, dataset provenance, and scientific caveats.

## Source Coverage

Local PDFs reviewed for this pass:

- `references/datasets_used/camelyon16.pdf`
- `references/datasets_used/catch.pdf`
- `references/datasets_used/diagset.pdf`
- `references/datasets_used/hiesd.pdf`
- `references/papers_used/1-s2.0-S2153353923001359-main.pdf`
- `references/papers_used/10278_2023_Article_814.pdf`
- `references/papers_used/A generalized deep learning.pdf`
- `references/papers_used/Grandqc.pdf`
- `references/papers_used/Int J Imaging Syst Tech - 2024 - Seghier - Image Segmentation Evaluation With the Dice Index  Methodological Issues.pdf`
- `references/papers_used/pyhist.pdf`
- `references/papers_used/s41551-026-01616-8.pdf`

Existing augmentation references that were already in this file were preserved
because they still support the training-augmentation section.

## Dataset Source Papers

| Dataset/source | Suggested key | Use in thesis/paper | Caveat |
| --- | --- | --- | --- |
| CAMELYON16 lymph-node metastasis challenge: Bejnordi et al. (2017) | `bejnordi2017diagnostic` | Cite for CAMELYON16 provenance, breast-cancer lymph-node WSI metastasis detection, challenge-style benchmark framing, and the presence of patient/slide-level diagnostic evaluation. | Do not cite it as a segmentation-only dataset paper; the article evaluates challenge algorithms and pathologist comparison. |
| CATCH canine cutaneous cancer dataset: Wilm et al. (2022) | `wilm2022catch` | Cite for CATCH provenance, canine cutaneous tumor WSIs, polygon annotations, and multi-class histology labels. | Mention species/domain shift if combining with human datasets. |
| DiagSet prostate histopathology dataset: Koziarski et al. (2024) | `koziarski2024diagset` | Cite for DiagSet provenance, prostate-cancer WSI/patch data, fully annotated scans, binary scan diagnoses, and patch-based cancer-detection framing. | Do not claim the repository reproduces the DiagSet model framework unless that experiment was run. |
| HiESD early gastric cancer and precancerous lesions dataset: Wang et al. (2025) | `wang2025hiesd` | Cite for HiESD provenance, ESD specimen pathology slides, detailed region-level annotations, and early gastric cancer/precancerous lesion categories. | Preserve the repository's hardcoded HIESD label policy separately from the paper citation. |

## Pipeline Decision Papers

| Pipeline decision | Suggested citations | How to use them |
| --- | --- | --- |
| Whole-slide quality control and artifact motivation | `weng2024grandqc` | Cite GrandQC when explaining why slide artifacts and tissue-quality masks matter before downstream computational pathology. It supports the Stage 1 artifact/QC motivation. |
| WSI preprocessing, tissue/background separation, and tile extraction | `munozaguirre2020pyhist`, `khened2021generalized` | Cite PyHIST for the general need to segment tissue and generate tiles from WSIs. Cite Khened et al. for patch-based WSI segmentation pipelines and the engineering pressure created by gigapixel slides. |
| Scanner and acquisition variability | `duenweg2023scanner` | Cite when discussing scanner-dependent optical variation, downstream feature sensitivity, and why robust preprocessing/provenance tracking matters. |
| Stain/color variability and runtime normalization motivation | `tellez2019augmentation`, `duenweg2023scanner` | Cite Tellez et al. for computational pathology stain/color variability and normalization/augmentation motivation. Cite Duenweg et al. for scanner-driven optical variability. |
| Online image augmentation during training | `buslaev2020albumentations`, `dossantos2023augmentation`, `tellez2019augmentation`, `shorten2019survey`, `zhong2020random`, `devries2017cutout` | Cite Albumentations for implementation, dos Santos et al. and Tellez et al. for histopathology-specific augmentation motivation, Shorten and Khoshgoftaar for the broad augmentation survey, and Random Erasing/Cutout for coarse-dropout-style occlusion. |
| Patch-based segmentation, ensemble inference, and WSI-scale constraints | `khened2021generalized` | Cite for a comparable WSI segmentation pipeline using patch-based processing, class-imbalance handling, ensemble segmentation, and efficient inference. |
| Stage 2 tissue filtering and patch extraction | `khened2021generalized`, `munozaguirre2020pyhist`, `bandi2019resolution` | Cite Khened et al. for HSV/Otsu/morphology tissue masking and patch extraction, PyHIST for tissue-content tile extraction, and Bandi et al. for the broader importance of tissue segmentation before WSI analysis. |
| Stage 2 artifact-aware extraction provenance | `weng2024grandqc` | Cite GrandQC to justify recording artifact/tissue-quality information because artifacts can harm downstream computational pathology models. |
| Stage 2 ambiguous-label rejection | `yu2020noisy` | Cite Yu et al. for the general principle that noisy annotations degrade medical image segmentation; use this to justify rejecting cancer/not-cancer overlaps and ambiguous windows. |
| Stage 2 manifest provenance and stale-input checks | `wilkinson2016fair` | Cite FAIR for rich metadata and provenance as a reproducibility principle; this supports processing signatures, source lineage, and manifest-backed reuse checks. |
| Stage 3.1 manual-review sampling and proportion sizing | `charan2013sample`, `yu2020noisy` | Cite Charan and Biswas for sample-size calculation around estimating a population proportion. Cite Yu et al. only for the motivation that manual review is guarding against noisy labels/contaminated annotations. |
| Stage 3.2 graph-contamination tuning | `felzenszwalb2004efficient`, `snoek2012practical`, `roberts2017crossvalidation`, `sokolova2009performance` | Cite Felzenszwalb and Huttenlocher for the graph-segmentation primitive, Snoek et al. for Bayesian hyperparameter optimization, Roberts et al. for grouped/structured cross-validation, and Sokolova and Lapalme for F1 as a classification performance measure. |
| Stage 3.3 manifest-driven graph cleaning | `felzenszwalb2004efficient`, `yu2020noisy`, `wilkinson2016fair` | Cite Felzenszwalb for the segmentation primitive, Yu et al. for label-noise motivation, and FAIR for preserving source lineage and decision manifests instead of rewriting canonical HDF5 shards. |
| Stage 4 patient-level split optimization | `roberts2017crossvalidation`, `dawood2026confounding`, `akiba2019optuna`, `bergstra2011algorithms` | Cite Roberts et al. and Dawood et al. for patient/structured split isolation and leakage/confounding caution. Cite Optuna and TPE papers for the optimization machinery; the exact greedy cancer-ratio allocator is repository-specific. |
| Stage 4 split balance verification and provenance | `pearson1900criterion`, `wilkinson2016fair` | Cite Pearson only for the chi-square statistic used as a raw split-balance verification value. Cite FAIR for manifest, run-config, split-stats, and SQLite-linked provenance. |
| Stage 4 entropy-guided template selection | `shannon1948mathematical`, `tellez2019augmentation` | Cite Shannon for the entropy measure and Tellez et al. for stain/color variability motivation. The highest-entropy-per-TRAIN-patient template choice is a repository heuristic. |
| Stage 4 runtime stain-normalization artifacts | `reinhard2001color`, `ruifrok2001quantification`, `macenko2009method`, `vahadane2016structure`, `tellez2019augmentation`, `duenweg2023scanner` | Cite the original method papers for the persisted runtime-normalizer state and Tellez/Duenweg for why stain/scanner variability matters. Stage 4 persists target/reference state; it does not normalize all split pixels eagerly. |
| Training loss implementation | `khened2021generalized` | Cite Khened et al. as the direct source for the weighted BCE-plus-Dice loss implemented in Stage 8. |
| Dice reporting and segmentation metric caveats | `seghier2024dice` | Cite when explaining Dice as an overlap metric, when warning that Dice is sensitive to reporting choices, and when justifying transparent metric definitions. |
| Confounding, leakage, and cautious interpretation in computational pathology | `dawood2026confounding` | Cite for the broader warning that histology models can learn confounded correlational signals. This supports patient-level splitting, provenance validation, and cautious claims. |

## Suggested Methods Integration

Use these as citation anchors when updating `reports/main.tex`.

- Stage 1 artifact/QC: cite GrandQC (`weng2024grandqc`) for why artifact and
  tissue-quality control can materially affect downstream pathology models.
- Stage 2 extraction: cite PyHIST (`munozaguirre2020pyhist`) and Khened et al.
  (`khened2021generalized`) for WSI tissue preprocessing, patching, and
  gigapixel-scale constraints. If describing the actual Stage 2 tissue filter,
  cite Khened et al. directly because the repository uses the same HSV
  saturation, Otsu thresholding, and morphology pattern.
- Stage 2 annotation filtering: cite Yu et al. (`yu2020noisy`) when explaining
  why cross-label cancer/not-cancer overlaps and ambiguous windows are rejected
  instead of being treated as usable labels.
- Stage 2 artifact-aware lineage: cite GrandQC (`weng2024grandqc`) when
  explaining per-patch artifact coverage and artifact-aware provenance.
- Stage 2 manifest provenance: cite FAIR (`wilkinson2016fair`) when explaining
  source-file lineage, processing signatures, stale-input detection, and
  manifest-backed reproducibility checks.
- Stage 3.1 optimization sampling: cite Charan and Biswas
  (`charan2013sample`) for the proportion-based sample-size calculation using
  confidence level, margin of error, and assumed class proportion. Cite Yu et al.
  (`yu2020noisy`) when explaining why the sampled overlays are manually reviewed
  to separate acceptable from rejected cancer-region candidates.
- Stage 3.2 graph-method tuning: cite Felzenszwalb and Huttenlocher
  (`felzenszwalb2004efficient`) for the graph segmentation used by
  `skimage.segmentation.felzenszwalb`, Snoek et al. (`snoek2012practical`) for
  Gaussian-process Bayesian optimization with expected improvement, Roberts et
  al. (`roberts2017crossvalidation`) for patient/slide-grouped validation of
  structured data, and Sokolova and Lapalme (`sokolova2009performance`) when
  explaining the rejected-class F1 objective.
- Stage 3.3 graph cleaning: cite Felzenszwalb and Huttenlocher
  (`felzenszwalb2004efficient`) for the segmentation primitive, Yu et al.
  (`yu2020noisy`) for the label-noise/contamination motivation, and FAIR
  (`wilkinson2016fair`) for persisting accepted/rejected decisions and source
  HDF5 lineage through the manifest rather than rewriting Stage 2 shards.
- Stage 4 split generation: cite Roberts et al.
  (`roberts2017crossvalidation`) and Dawood et al. (`dawood2026confounding`)
  for patient/structured split isolation and leakage/confounding caution. Cite
  Optuna (`akiba2019optuna`) and TPE (`bergstra2011algorithms`) for the
  search machinery, but describe the objective as the repository's own
  patient-level greedy allocator minimizing class-ratio deviation.
- Stage 4 split diagnostics and provenance: cite Pearson
  (`pearson1900criterion`) only when explaining the raw chi-square split-balance
  verification statistic, and cite FAIR (`wilkinson2016fair`) for
  manifest-backed split assignments, run configuration, runtime environment, and
  SQLite-linked split-bundle artifacts.
- Stage 4 template and normalization artifacts: cite Shannon
  (`shannon1948mathematical`) for grayscale entropy, Tellez et al.
  (`tellez2019augmentation`) and Duenweg et al. (`duenweg2023scanner`) for
  stain/scanner variability, and the method papers for the runtime-normalizer
  states: Reinhard (`reinhard2001color`), Ruifrok
  (`ruifrok2001quantification`), Macenko (`macenko2009method`), and Vahadane
  (`vahadane2016structure`).
- Dataset provenance: cite the dataset paper associated with each active
  dataset cohort: CAMELYON16 (`bejnordi2017diagnostic`), CATCH
  (`wilm2022catch`), DiagSet (`koziarski2024diagset`), and HiESD
  (`wang2025hiesd`).
- Stage 4 and downstream normalization: cite Tellez et al.
  (`tellez2019augmentation`) for stain normalization/color variability and
  Duenweg et al. (`duenweg2023scanner`) for scanner-dependent optical
  differences.
- Stages 7 and 8 augmentation: cite Albumentations
  (`buslaev2020albumentations`) for the library and dos Santos et al.
  (`dossantos2023augmentation`) for histological segmentation augmentation.
  Use `zhong2020random` and `devries2017cutout` only for the coarse
  dropout/occlusion analogy.
- Stage 8 training loss: cite Khened et al. (`khened2021generalized`) because
  the repository loss is a direct implementation of their weighted
  BCE-plus-Dice loss.
- Stage 8 metrics and Stage 10 reporting: cite Seghier
  (`seghier2024dice`) when defining Dice and warning against opaque Dice
  comparisons.
- Scientific limitations: cite Dawood et al. (`dawood2026confounding`) for
  confounding/bias caution when interpreting model behavior across pathology
  cohorts.

## Scientific Wording Caveats

- Do not state that the repository reproduces GrandQC, PyHIST, or the full
  Khened et al. framework. The pipeline uses related motivations and design
  patterns from these works; the Stage 8 loss is the direct Khened et al.
  implementation-specific exception.
- For Stage 2, cite Khened et al. as a direct match only for the tissue-mask
  extraction pattern and as supporting context for patch-based WSI processing.
  Do not claim the full Khened training/inference framework is reproduced by
  Stage 2.
- Do not claim the Stage 2 overlap-removal rule comes from Yu et al. That paper
  supports the risk of noisy annotations in segmentation; the exact positive-area
  polygon rejection rule is repository-specific.
- Do not cite FAIR as if it specifies SQLite, HDF5, or the repository's manifest
  schema. FAIR supports the metadata/provenance rationale only.
- Do not claim Stage 3.1 validates final label quality by itself. Charan and
  Biswas support the sample-size calculation, while the actual manual
  approved/rejected criteria are repository-specific.
- Do not present the Stage 3.2/3.3 contamination score as a published
  Felzenszwalb method. The citation supports the graph segmentation primitive;
  background-segment classification, ROI erosion, tau selection, and the
  accept/reject rule are repository-specific heuristics.
- Do not claim Bayesian optimization proves clinical validity. Snoek et al.
  support the hyperparameter-search strategy only.
- Do not cite Roberts et al. as pathology-specific evidence. Use it to justify
  grouped validation for structured/non-independent observations, then describe
  the repository's grouping key explicitly.
- Do not cite Sokolova and Lapalme as if they define the repository threshold
  rule. They support reporting/optimizing F1 as a classification performance
  measure; the rejected-class threshold sweep is local pipeline behavior.
- Do not claim the Stage 4 patient allocator is a published split algorithm.
  The citations support patient/group isolation and hyperparameter-search
  machinery, while the weighted ordering plus greedy class-ratio objective is
  repository-specific.
- Do not interpret the Stage 4 chi-square values as formal hypothesis tests
  unless p-values, assumptions, and degrees of freedom are explicitly defined.
  The current code stores raw verification statistics only.
- Do not claim that Shannon entropy proves the selected templates are optimal
  stain references. Shannon supports the entropy definition; highest-entropy
  TRAIN-patient template selection is a heuristic.
- Do not claim that Stage 4 writes normalized TRAIN/VALIDATION/TEST pixel
  datasets. Current Stage 4 manifests are `NOT_NORMALIZED`; the stage persists
  runtime target/reference artifacts that later stages consume on the fly.
- Do not cite Reinhard, Ruifrok, Macenko, or Vahadane as if the repository
  exactly reproduces every implementation detail of the original papers. Cite
  them for the normalization/deconvolution families and then describe the actual
  `torch-staintools`/fixed-matrix runtime contract.
- Do not describe the current coarse dropout as exact Random Erasing or exact
  Cutout. It is an Albumentations `CoarseDropout` implementation with a related
  occlusion-regularization rationale.
- Do not describe the default `RUNTIME_VAHADANE_BACKEND=fixed_source` behavior
  as exact per-patch VAHADANE. It is a fixed-matrix approximation.
- Do not use dataset papers as evidence of the repository's final experimental
  performance. They support dataset provenance and task framing only.
- Do not compare Dice scores across datasets or preprocessing regimes without
  defining the averaging unit, empty-mask handling, thresholding, and whether
  the score is pixel-, patch-, slide-, or patient-level.

## Dataset Paper Notes

### CAMELYON16

Suggested key: `bejnordi2017diagnostic`

The paper describes the CAMELYON16 challenge for detecting lymph-node
metastases in H&E-stained breast-cancer sentinel lymph-node slides. The local
PDF reports a training set with 110 positive and 160 negative WSIs and an
independent test set of 129 WSIs. Use this citation for the origin and clinical
framing of CAMELYON16.

### CATCH

Suggested key: `wilm2022catch`

The paper presents the pan-tumor Canine cuTaneous Cancer Histology dataset,
with 350 WSIs, seven canine cutaneous tumor subtypes, 12,424 polygon
annotations, and 13 histologic classes. Use this citation for CATCH provenance
and multi-class polygon-annotation context.

### DiagSet

Suggested key: `koziarski2024diagset`

The paper introduces a prostate-cancer histopathology dataset with over
2.6 million tissue patches from 430 fully annotated scans, 4675 scans with
binary diagnoses, and 46 scans independently diagnosed by histopathologists.
Use it for DiagSet provenance and the patch/scan-level prostate-cancer task
framing.

### HiESD

Suggested key: `wang2025hiesd`

The paper presents a fully annotated ESD pathology-slide dataset for early
gastric cancer and precancerous lesions, comprising 308 de-identified tissue
samples from 104 H&E-stained digital slides and 10 region-level categories. Use
it for HiESD provenance and the early-gastric-cancer lesion-map framing.

## Stage 2 Database Manager Notes

Suggested code boundary: `2_database_manager.py`, `helpers/extraction/config.py`,
`helpers/extraction/data_handlers.py`, `helpers/extraction/patch_engine.py`,
`helpers/extraction/hdf5_storage.py`, and
`helpers/extraction/master_manifest.py`.

| Stage 2 decision | Code behavior to document | Suggested citation |
| --- | --- | --- |
| Patch extraction from annotated WSI regions | Stage 2 builds a grid of fixed-size windows, uses stride-controlled overlap, reads RGB patches from OpenSlide, and persists raw patch/mask/label rows in HDF5. | `khened2021generalized`; `munozaguirre2020pyhist` |
| Tissue-content filtering | Stage 2 rejects low-tissue patches using an HSV saturation mask, Otsu thresholding, and morphology before saving patches. | `khened2021generalized`; optionally `bandi2019resolution` for the general need for WSI tissue segmentation |
| Ambiguous annotation handling | Stage 2 removes whole cancer and not-cancer polygons with positive-area cross-label overlap, then skips windows whose cancer/not-cancer overlap does not meet the exclusive label rule. | `yu2020noisy` as general label-noise motivation; exact rule is repository-specific |
| Artifact-aware extraction lineage | When artifact GeoJSON is available, Stage 2 records per-patch coverage for artifact classes such as folds, pen markings, out-of-focus regions, dark spots/foreign objects, and edge/air-bubble regions. | `weng2024grandqc` |
| Dataset-specific annotation semantics | Stage 2 maps HIESD, CHILE, JSON, and NDPI/NDPA annotations into the repository's binary cancer/not-cancer contract, including HIESD XML coordinate scaling to level 0. | Dataset papers such as `wang2025hiesd` and `koziarski2024diagset`; cite only the active dataset source |
| Manifest-backed provenance | Stage 2 stores source slide, annotation, artifact GeoJSON, HDF5 row, source signature, processing signature, and status metadata in `master_manifest.sqlite`, and fails closed on stale inputs or manifest-integrity errors. | `wilkinson2016fair` |

Operational settings such as local slide staging, worker count, HDF5 compression,
OpenSlide cache size, and native-warning suppression should be reported for
reproducibility when they affect a run, but they do not need peer-reviewed
method citations unless the manuscript makes a scientific claim about them.

## Stage 3 Optimization And Graph Cleaning Notes

Suggested code boundary: `3_1_optimization_sampling.py`,
`3_2_tune_graph_method.py`, `3_3_cleaner_script.py`,
`helpers/optimization_sampling/*`, and `helpers/graph/*`.

| Stage 3 decision | Code behavior to document | Suggested citation |
| --- | --- | --- |
| Statistical manual-review sampling | Stage 3.1 calculates a required sample size for a proportion using confidence level, margin of error, and assumed proportion; defaults are 95% confidence, 5% margin of error, and 0.5 proportion. | `charan2013sample` |
| Non-overlapping pilot and master candidate pools | Stage 3.1 shuffles canonical cancer image/mask pairs from the Stage 2 manifest with a fixed seed and creates separate pilot and master-pool overlay sets. | Cite `charan2013sample` only for the sample-size logic; the non-overlap and seed policy are repository-specific reproducibility choices. |
| Manual overlay review for contamination/noise control | Stage 3.1 writes overlay images into `pilot_sample` and `master_candidate_pool`, each with `APPROVED` and `REJECTED` folders for operator review. | `yu2020noisy` as general label-noise motivation; overlay color/thickness/alpha are implementation details. |
| Manifest-backed source discovery | Stage 3.1 and 3.2 resolve image/mask pairs from `master_manifest.sqlite` instead of rediscovering files ad hoc, and require unique filename stems among canonical cancer rows. | `wilkinson2016fair` |
| Graph-based contamination score | Stage 3.2 and 3.3 use Felzenszwalb graph segmentation, classify bright/background segments by mean intensity, erode the ROI mask as a safety margin, and compute contamination as background pixels inside the ROI divided by eroded ROI area. | `felzenszwalb2004efficient`; the background-intensity, erosion, and contamination-ratio heuristic is repository-specific. |
| Bayesian tuning of graph parameters | Stage 3.2 tunes background-intensity threshold, Felzenszwalb scale `k`, minimum segment size, and ROI erosion with `gp_minimize`, expected improvement, configured random seed, and an evaluation budget. | `snoek2012practical` |
| Patient/slide-grouped validation | Stage 3.2 maps reviewed overlays back to source records, infers a group id from the stem, uses stratified grouped train/test and inner CV splits, and optimizes validation performance without mixing the same group across folds. | `roberts2017crossvalidation`; optionally `dawood2026confounding` when tying this to pathology confounding/leakage risk. |
| Rejected-class F1 threshold selection | Stage 3.2 scores contamination rates, sweeps tau from 0.05 to 0.95, and selects the threshold that maximizes F1 for the `Rejected` class before reporting held-out accuracy/F1. | `sokolova2009performance` |
| Parameter artifact as Stage 3.2 to 3.3 contract | Stage 3.2 writes a JSON artifact containing graph parameters, tau, cross-validated F1, data counts, random seed, and generator script; Stage 3.3 requires this artifact through `GRAPH_CLEANING_PARAMS_PATH`. | `wilkinson2016fair` |
| Manifest-driven graph cleaning without rewriting Stage 2 shards | Stage 3.3 scores only cancer rows, accepts non-cancer rows without graph scoring, stores decisions in `patch_stage_state`, writes accepted/rejected CSV manifests with source lineage, and treats Stage 2 HDF5 shards as read-only. | `yu2020noisy` for contamination/noise motivation and `wilkinson2016fair` for lineage/reproducibility. |

## Stage 4 Crossfold Notes

Suggested code boundary: `4_crossfold.py`, `helpers/crossfold/config.py`,
`helpers/crossfold/discovery.py`, `helpers/crossfold/splitting.py`,
`helpers/crossfold/entropy.py`, `helpers/crossfold/normalization.py`, and
`helpers/crossfold/provenance.py`.

| Stage 4 decision | Code behavior to document | Suggested citation |
| --- | --- | --- |
| SQLite manifest as source contract | Stage 4 currently requires `CROSSFOLD_SOURCE_HDF5_PATH` to point to `master_manifest.sqlite` and loads only rows where Stage 3.3 marked `is_stage4_accepted = 1`. | `wilkinson2016fair` |
| Patient-level TRAIN/VALIDATION/TEST assignment | Stage 4 builds one row per patient, keeps all patches from a patient in one split, requires requested TEST and VALIDATION patient counts, and rejects held-out splits containing only one class. | `roberts2017crossvalidation`; `dawood2026confounding` for pathology leakage/confounding caution |
| Optuna/TPE-guided greedy split search | Stage 4 gives each patient a trial weight, orders patients by Optuna's TPE sampler, greedily assigns patients with TEST/VALIDATION priority, and minimizes the sum of absolute cancer-ratio deviations across TRAIN, VALIDATION, and TEST. | `akiba2019optuna`; `bergstra2011algorithms`; exact allocator and objective are repository-specific |
| Split-balance verification | Stage 4 records expected cancer/non-cancer counts and a raw chi-square statistic per split against the global cancer ratio. | `pearson1900criterion`; report as verification, not as an inferential p-value unless additional assumptions are documented |
| TRAIN-only entropy template selection | Stage 4 computes grayscale Shannon entropy on TRAIN patches, selects the highest-entropy TRAIN patch per patient, and saves template provenance. | `shannon1948mathematical`; selection heuristic is repository-specific |
| Aggregate target construction | Stage 4 builds a median RGB aggregate target from the selected TRAIN templates and saves `aggregate_target.png`. | `tellez2019augmentation`; `duenweg2023scanner`; cite as stain/scanner variability motivation, not proof that the aggregate target is optimal |
| Runtime-normalization artifact generation | Stage 4 fits and saves JSON runtime state for REINHARD, RUIFROK, MACENKO, and VAHADANE under `runtime_normalization_artifacts/<method>/normalization_stats.json`, then links each artifact to the split bundle in SQLite. | `reinhard2001color`; `ruifrok2001quantification`; `macenko2009method`; `vahadane2016structure`; `wilkinson2016fair` for the artifact linkage |
| Normalization-agnostic split outputs | Stage 4 writes split manifests as `NOT_NORMALIZED` and persists runtime artifacts for later on-the-fly normalization; it does not eagerly rewrite every split row into normalized pixel storage. | `tellez2019augmentation` for normalization motivation; implementation contract is repository-specific |
| Reproducibility artifacts | Stage 4 writes `manifest.csv`, `split_stats.csv`, `run_config.json`, template-selection metadata, runtime environment versions, git commit, and SQLite split assignments. | `wilkinson2016fair` |

## BibTeX

```bibtex
@article{bejnordi2017diagnostic,
  title = {Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer},
  author = {Ehteshami Bejnordi, Babak and Veta, Mitko and van Diest, Paul Johannes and van Ginneken, Bram and Karssemeijer, Nico and Litjens, Geert and van der Laak, Jeroen A. W. M. and {CAMELYON16 Consortium}},
  journal = {JAMA},
  volume = {318},
  number = {22},
  pages = {2199--2210},
  year = {2017},
  doi = {10.1001/jama.2017.14585},
  url = {https://doi.org/10.1001/jama.2017.14585}
}

@article{wilm2022catch,
  title = {Pan-tumor CAnine cuTaneous Cancer Histology (CATCH) dataset},
  author = {Wilm, Frauke and Fragoso, Marco and Marzahl, Christian and Qiu, Jingna and Puget, Chlo{\'e} and Diehl, Laura and Bertram, Christof A. and Klopfleisch, Robert and Maier, Andreas and Breininger, Katharina and Aubreville, Marc},
  journal = {Scientific Data},
  volume = {9},
  number = {1},
  pages = {588},
  year = {2022},
  doi = {10.1038/s41597-022-01692-w},
  url = {https://doi.org/10.1038/s41597-022-01692-w}
}

@article{koziarski2024diagset,
  title = {DiagSet: a dataset for prostate cancer histopathological image classification},
  author = {Koziarski, Micha{\l} and Cyganek, Bogus{\l}aw and Niedziela, Przemys{\l}aw and Olborski, Bogus{\l}aw and Antosz, Zbigniew and {\.Z}ydak, Marcin and Kwolek, Bogdan and W{\k a}sowicz, Pawe{\l} and Buka{\l}a, Andrzej and Swad{\'z}ba, Jakub and Sitkowski, Piotr},
  journal = {Scientific Reports},
  volume = {14},
  number = {1},
  pages = {6780},
  year = {2024},
  doi = {10.1038/s41598-024-52183-4},
  url = {https://doi.org/10.1038/s41598-024-52183-4}
}

@article{wang2025hiesd,
  title = {A fully annotated pathology slide dataset for early gastric cancer and precancerous lesions},
  author = {Wang, Chunbao and Ge, Jiusong and Niu, Yi and Ding, Caixia and Fan, Yangyang and Chang, Hongyun and Yang, Zhe and Ran, Caihong and Teng, Xiali and Wang, Xiaolin and Wu, Lianlian and Gao, Zeyu and Li, Chen},
  journal = {Scientific Data},
  volume = {12},
  number = {1},
  pages = {1326},
  year = {2025},
  doi = {10.1038/s41597-025-05679-1},
  url = {https://doi.org/10.1038/s41597-025-05679-1}
}

@article{duenweg2023scanner,
  title = {Whole slide imaging (WSI) scanner differences influence optical and computed properties of digitized prostate cancer histology},
  author = {Duenweg, Savannah R. and Bobholz, Samuel A. and Lowman, Allison K. and Stebbins, Margaret A. and Winiarz, Aleksandra and Nath, Biprojit and Kyereme, Fitzgerald and Iczkowski, Kenneth A. and LaViolette, Peter S.},
  journal = {Journal of Pathology Informatics},
  volume = {14},
  pages = {100321},
  year = {2023},
  doi = {10.1016/j.jpi.2023.100321},
  url = {https://doi.org/10.1016/j.jpi.2023.100321}
}

@article{dossantos2023augmentation,
  title = {Influence of Data Augmentation Strategies on the Segmentation of Oral Histological Images Using Fully Convolutional Neural Networks},
  author = {dos Santos, Dal{\'i} F. D. and de Faria, Paulo R. and Traven{\c c}olo, Bruno A. N. and do Nascimento, Marcelo Z.},
  journal = {Journal of Digital Imaging},
  volume = {36},
  number = {4},
  pages = {1608--1623},
  year = {2023},
  doi = {10.1007/s10278-023-00814-z},
  url = {https://doi.org/10.1007/s10278-023-00814-z}
}

@article{khened2021generalized,
  title = {A generalized deep learning framework for whole-slide image segmentation and analysis},
  author = {Khened, Mahendra and Kori, Avinash and Rajkumar, Haran and Krishnamurthi, Ganapathy and Srinivasan, Balaji},
  journal = {Scientific Reports},
  volume = {11},
  number = {1},
  pages = {11579},
  year = {2021},
  doi = {10.1038/s41598-021-90444-8},
  url = {https://doi.org/10.1038/s41598-021-90444-8}
}

@article{bandi2019resolution,
  title = {Resolution-agnostic tissue segmentation in whole-slide histopathology images with convolutional neural networks},
  author = {B{\'a}ndi, P{\'e}ter and Balkenhol, Maschenka and van Ginneken, Bram and van der Laak, Jeroen and Litjens, Geert},
  journal = {PeerJ},
  volume = {7},
  pages = {e8242},
  year = {2019},
  doi = {10.7717/peerj.8242},
  url = {https://doi.org/10.7717/peerj.8242}
}

@article{weng2024grandqc,
  title = {GrandQC: A comprehensive solution to quality control problem in digital pathology},
  author = {Weng, Zhilong and Seper, Alexander and Pryalukhin, Alexey and Mairinger, Fabian and Wickenhauser, Claudia and Bauer, Marcus and Glamann, Lennert and Bl{\"a}ker, Hendrik and Lingscheidt, Thomas and Hulla, Wolfgang and Jonigk, Danny and Schallenberg, Simon and Bychkov, Andrey and Fukuoka, Junya and Braun, Martin and Sch{\"o}mig-Markiefka, Birgid and Klein, Sebastian and Thiel, Andreas and Bozek, Katarzyna and Netto, George J. and Quaas, Alexander and B{\"u}ttner, Reinhard and Tolkach, Yuri},
  journal = {Nature Communications},
  volume = {15},
  number = {1},
  pages = {10685},
  year = {2024},
  doi = {10.1038/s41467-024-54769-y},
  url = {https://doi.org/10.1038/s41467-024-54769-y}
}

@article{seghier2024dice,
  title = {Image Segmentation Evaluation With the Dice Index: Methodological Issues},
  author = {Seghier, Mohamed L.},
  journal = {International Journal of Imaging Systems and Technology},
  volume = {34},
  number = {6},
  pages = {e23203},
  year = {2024},
  doi = {10.1002/ima.23203},
  url = {https://doi.org/10.1002/ima.23203}
}

@article{munozaguirre2020pyhist,
  title = {PyHIST: A Histological Image Segmentation Tool},
  author = {Mu{\~n}oz-Aguirre, Manuel and Ntasis, Vasilis F. and Rojas, Santiago and Guig{\'o}, Roderic},
  journal = {PLOS Computational Biology},
  volume = {16},
  number = {10},
  pages = {e1008349},
  year = {2020},
  doi = {10.1371/journal.pcbi.1008349},
  url = {https://doi.org/10.1371/journal.pcbi.1008349}
}

@article{yu2020noisy,
  title = {Robustness study of noisy annotation in deep learning based medical image segmentation},
  author = {Yu, Shaode and Chen, Mingli and Zhang, Erlei and Wu, Junjie and Yu, Hang and Yang, Zi and Ma, Lin and Gu, Xuejun and Lu, Weiguo},
  journal = {Physics in Medicine \& Biology},
  volume = {65},
  number = {17},
  pages = {175007},
  year = {2020},
  doi = {10.1088/1361-6560/ab99e5},
  url = {https://doi.org/10.1088/1361-6560/ab99e5}
}

@article{wilkinson2016fair,
  title = {The FAIR Guiding Principles for scientific data management and stewardship},
  author = {Wilkinson, Mark D. and Dumontier, Michel and Aalbersberg, IJsbrand Jan and Appleton, Gabrielle and Axton, Myles and Baak, Arie and Blomberg, Niklas and Boiten, Jan-Willem and da Silva Santos, Luiz Bonino and Bourne, Philip E. and Bouwman, Jildau and Brookes, Anthony J. and Clark, Tim and Crosas, Merc{\`e} and Dillo, Ingrid and Dumon, Olivier and Edmunds, Scott and Evelo, Chris T. and Finkers, Richard and Gonzalez-Beltran, Alejandra and Gray, Alasdair J. G. and Groth, Paul and Goble, Carole and Grethe, Jeffrey S. and Heringa, Jaap and {t Hoen}, Peter A. C. and Hooft, Rob and Kuhn, Tobias and Kok, Ruben and Kok, Joost and Lusher, Scott J. and Martone, Maryann E. and Mons, Albert and Packer, Abel L. and Persson, Bengt and Rocca-Serra, Philippe and Roos, Marco and van Schaik, Rene and Sansone, Susanna-Assunta and Schultes, Erik and Sengstag, Thierry and Slater, Ted and Strawn, George and Swertz, Morris A. and Thompson, Mark and van der Lei, Johan and van Mulligen, Erik and Velterop, Jan and Waagmeester, Andra and Wittenburg, Peter and Wolstencroft, Katherine and Zhao, Jun and Mons, Barend},
  journal = {Scientific Data},
  volume = {3},
  number = {1},
  pages = {160018},
  year = {2016},
  doi = {10.1038/sdata.2016.18},
  url = {https://doi.org/10.1038/sdata.2016.18}
}

@article{charan2013sample,
  title = {How to Calculate Sample Size for Different Study Designs in Medical Research?},
  author = {Charan, Jaykaran and Biswas, Tamoghna},
  journal = {Indian Journal of Psychological Medicine},
  volume = {35},
  number = {2},
  pages = {121--126},
  year = {2013},
  doi = {10.4103/0253-7176.116232},
  url = {https://doi.org/10.4103/0253-7176.116232}
}

@article{felzenszwalb2004efficient,
  title = {Efficient Graph-Based Image Segmentation},
  author = {Felzenszwalb, Pedro F. and Huttenlocher, Daniel P.},
  journal = {International Journal of Computer Vision},
  volume = {59},
  number = {2},
  pages = {167--181},
  year = {2004},
  doi = {10.1023/B:VISI.0000022288.19776.77},
  url = {https://doi.org/10.1023/B:VISI.0000022288.19776.77}
}

@inproceedings{snoek2012practical,
  title = {Practical Bayesian Optimization of Machine Learning Algorithms},
  author = {Snoek, Jasper and Larochelle, Hugo and Adams, Ryan P.},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {25},
  year = {2012},
  url = {https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms}
}

@article{roberts2017crossvalidation,
  title = {Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure},
  author = {Roberts, David R. and Bahn, Volker and Ciuti, Simone and Boyce, Mark S. and Elith, Jane and Guillera Arroita, Gurutzeta and Hauenstein, Severin and Lahoz-Monfort, Jos{\'e} J. and Schr{\"o}der, Boris and Thuiller, Wilfried and Warton, David I. and Wintle, Brendan A. and Hartig, Florian and Dormann, Carsten F.},
  journal = {Ecography},
  volume = {40},
  number = {8},
  pages = {913--929},
  year = {2017},
  doi = {10.1111/ecog.02881},
  url = {https://doi.org/10.1111/ecog.02881}
}

@article{sokolova2009performance,
  title = {A systematic analysis of performance measures for classification tasks},
  author = {Sokolova, Marina and Lapalme, Guy},
  journal = {Information Processing \& Management},
  volume = {45},
  number = {4},
  pages = {427--437},
  year = {2009},
  doi = {10.1016/j.ipm.2009.03.002},
  url = {https://doi.org/10.1016/j.ipm.2009.03.002}
}

@inproceedings{akiba2019optuna,
  title = {Optuna: A Next-generation Hyperparameter Optimization Framework},
  author = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages = {2623--2631},
  year = {2019},
  doi = {10.1145/3292500.3330701},
  url = {https://doi.org/10.1145/3292500.3330701}
}

@inproceedings{bergstra2011algorithms,
  title = {Algorithms for Hyper-Parameter Optimization},
  author = {Bergstra, James S. and Bardenet, R{\'e}mi and Bengio, Yoshua and K{\'e}gl, Bal{\'a}zs},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {24},
  pages = {2546--2554},
  year = {2011},
  url = {https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization}
}

@article{pearson1900criterion,
  title = {On the Criterion that a Given System of Deviations from the Probable in the Case of a Correlated System of Variables is Such that it Can be Reasonably Supposed to Have Arisen from Random Sampling},
  author = {Pearson, Karl},
  journal = {The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science},
  volume = {50},
  number = {302},
  pages = {157--175},
  year = {1900},
  doi = {10.1080/14786440009463897},
  url = {https://doi.org/10.1080/14786440009463897}
}

@article{shannon1948mathematical,
  title = {A Mathematical Theory of Communication},
  author = {Shannon, Claude E.},
  journal = {The Bell System Technical Journal},
  volume = {27},
  pages = {379--423, 623--656},
  year = {1948},
  url = {https://pure.mpg.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_2383162}
}

@article{reinhard2001color,
  title = {Color Transfer Between Images},
  author = {Reinhard, Erik and Ashikhmin, Michael and Gooch, Bruce and Shirley, Peter},
  journal = {IEEE Computer Graphics and Applications},
  volume = {21},
  number = {5},
  pages = {34--41},
  year = {2001},
  doi = {10.1109/38.946629},
  url = {https://doi.org/10.1109/38.946629}
}

@article{ruifrok2001quantification,
  title = {Quantification of Histochemical Staining by Color Deconvolution},
  author = {Ruifrok, Arnout C. and Johnston, Dennis A.},
  journal = {Analytical and Quantitative Cytology and Histology},
  volume = {23},
  number = {4},
  pages = {291--299},
  year = {2001},
  pmid = {11815294},
  url = {https://pubmed.ncbi.nlm.nih.gov/11815294/}
}

@inproceedings{macenko2009method,
  title = {A Method for Normalizing Histology Slides for Quantitative Analysis},
  author = {Macenko, Marc and Niethammer, Marc and Marron, J. S. and Borland, David and Woosley, John T. and Guan, Xiaojun and Schmitt, Charles and Thomas, Nancy E.},
  booktitle = {2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro},
  pages = {1107--1110},
  year = {2009},
  doi = {10.1109/ISBI.2009.5193250},
  url = {https://doi.org/10.1109/ISBI.2009.5193250}
}

@article{vahadane2016structure,
  title = {Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images},
  author = {Vahadane, Abhishek and Peng, Tingying and Sethi, Amit and Albarqouni, Shadi and Wang, Lichao and Baust, Maximilian and Steiger, Katja and Schlitter, Anna Melissa and Esposito, Irene and Navab, Nassir},
  journal = {IEEE Transactions on Medical Imaging},
  volume = {35},
  number = {8},
  pages = {1962--1971},
  year = {2016},
  doi = {10.1109/TMI.2016.2529665},
  url = {https://doi.org/10.1109/TMI.2016.2529665}
}

@article{dawood2026confounding,
  title = {Confounding factors and biases abound when predicting molecular biomarkers from histological images},
  author = {Dawood, Muhammad and Branson, Kim and Tejpar, Sabine and Rajpoot, Nasir and Minhas, Fayyaz ul Amir Afsar},
  journal = {Nature Biomedical Engineering},
  year = {2026},
  doi = {10.1038/s41551-026-01616-8},
  url = {https://doi.org/10.1038/s41551-026-01616-8},
  note = {Advance online publication}
}

@article{buslaev2020albumentations,
  title = {Albumentations: Fast and Flexible Image Augmentations},
  author = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
  journal = {Information},
  volume = {11},
  number = {2},
  pages = {125},
  year = {2020},
  doi = {10.3390/info11020125},
  url = {https://doi.org/10.3390/info11020125}
}

@article{tellez2019augmentation,
  title = {Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology},
  author = {Tellez, David and Litjens, Geert and B{\'a}ndi, P{\'e}ter and Bulten, Wouter and Bokhorst, John-Melle and Ciompi, Francesco and van der Laak, Jeroen},
  journal = {Medical Image Analysis},
  volume = {58},
  pages = {101544},
  year = {2019},
  doi = {10.1016/j.media.2019.101544},
  url = {https://doi.org/10.1016/j.media.2019.101544}
}

@article{shorten2019survey,
  title = {A survey on Image Data Augmentation for Deep Learning},
  author = {Shorten, Connor and Khoshgoftaar, Taghi M.},
  journal = {Journal of Big Data},
  volume = {6},
  number = {1},
  pages = {60},
  year = {2019},
  doi = {10.1186/s40537-019-0197-0},
  url = {https://doi.org/10.1186/s40537-019-0197-0}
}

@inproceedings{zhong2020random,
  title = {Random Erasing Data Augmentation},
  author = {Zhong, Zhun and Zheng, Liang and Kang, Guoliang and Li, Shaozi and Yang, Yi},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {34},
  number = {7},
  pages = {13001--13008},
  year = {2020},
  doi = {10.1609/aaai.v34i07.7000},
  url = {https://doi.org/10.1609/aaai.v34i07.7000}
}

@article{devries2017cutout,
  title = {Improved Regularization of Convolutional Neural Networks with Cutout},
  author = {DeVries, Terrance and Taylor, Graham W.},
  journal = {arXiv preprint arXiv:1708.04552},
  year = {2017},
  url = {https://arxiv.org/abs/1708.04552}
}
```
