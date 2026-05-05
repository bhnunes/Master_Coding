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
| Stage 5 fail-closed sanity gate and provenance consistency | `piccolo2016tools`, `wilkinson2016fair` | Cite Piccolo and Frampton for computational reproducibility practices around preserving executable workflows, software context, and analysis traceability. Cite FAIR for rich metadata/provenance. The exact checks and verdict names are repository-specific. |
| Stage 5 split, storage, and label-integrity checks | `roberts2017crossvalidation`, `dawood2026confounding`, `folk2011hdf5`, `yu2020noisy`, `he2009imbalanced` | Cite Roberts/Dawood for patient-level leakage and structured-data caution, Folk et al. for HDF5 as the storage substrate, Yu et al. for noisy annotation/mask-label risks, and He/Garcia for documenting class imbalance. |
| Stage 6 embedding-based smart sampling | `campanella2025clinical`, `oquab2024dinov2`, `he2009imbalanced` | Cite Campanella et al. for the clinical benchmarking context around public self-supervised pathology foundation models, Oquab et al. for DINOv2-style self-supervised visual features, and He/Garcia for the class-imbalance rationale behind protecting positive rows before sampling. |
| Stage 6 stability, clustering, and optional GIST-style selection | `hubert1985comparing`, `sculley2010webscale`, `fahrbach2025gist` | Cite Hubert and Arabie for the adjusted Rand index stability score, Sculley for MiniBatchKMeans, and Fahrbach et al. for GIST. The repository's adaptive budget, cluster-balanced order, and large-pool landmark preselection are implementation-specific. |
| Stage 6 compact TRAIN_SELECTED storage and sampling provenance | `folk2011hdf5`, `wilkinson2016fair`, `piccolo2016tools` | Cite HDF5 for compact selected-TRAIN shards and FAIR/Piccolo for preserving sidecars, run configuration, row mappings, summary JSON, and manifest-backed sampling decisions. |
| Stage 7 learning-rate and loss-weight screening | `smith2017cyclical`, `mckay1979comparison`, `khened2021generalized`, `loshchilov2019decoupled` | Cite Smith for the learning-rate range-test idea, McKay et al. for Latin-hypercube sampling of loss weights, Khened et al. for the weighted BCE-plus-Dice loss family, and AdamW for the optimizer. The repository's median-minimum-loss ranking is a screening heuristic. |
| Stage 7 runtime, precision, and data-provenance controls | `micikevicius2018mixed`, `he2009imbalanced`, `folk2011hdf5`, `wilkinson2016fair`, `piccolo2016tools` | Cite mixed-precision training only when AMP is enabled or discussed, He/Garcia for weighted sampling under class imbalance, HDF5 for compact TRAIN_SELECTED reads, and FAIR/Piccolo for run configuration, runtime environment, and provenance sidecars. |
| Stage 8 registered segmentation architectures and encoders | `ronneberger2015unet`, `liu2021swin`, `chen2018deeplabv3plus`, `zhang2022resnest`, `zhou2020unetpp`, `tan2019efficientnet`, `lin2017fpn`, `hu2018senet`, `xie2021segformer`, `fan2020manet`, `he2016resnet`, `ranftl2021dpt`, `dosovitskiy2021vit`, `xiao2018upernet`, `ryali2023hiera`, `russakovsky2015imagenet` | Cite the original decoder and encoder/backbone papers for the active registry pairs. The registry gates approved model/encoder combinations; the exact implementation is through `segmentation_models_pytorch` and `timm`, not a from-scratch reimplementation. |
| Stage 8 training loop, optimization, and validation selection | `khened2021generalized`, `loshchilov2019decoupled`, `defazio2024road`, `micikevicius2018mixed`, `shrivastava2016ohem`, `prechelt1998automatic`, `saito2015precision`, `matthews1975comparison`, `he2009imbalanced` | Cite Khened for the implemented BCE-plus-Dice loss, AdamW/Schedule-Free for the optimizer families, mixed precision when AMP is enabled, OHEM only when `TRAINING_RUN_OHEM=True`, early stopping for validation-based stopping, AUPRC/MCC papers for validation metrics, and He/Garcia for inverse-frequency sampling. |
| Stage 8 manifest-backed training provenance and compact TRAIN_SELECTED reads | `wilkinson2016fair`, `piccolo2016tools`, `folk2011hdf5`, `roberts2017crossvalidation`, `dawood2026confounding` | Cite FAIR/Piccolo for executable provenance and metadata sidecars, HDF5 for canonical and compact row storage, and Roberts/Dawood for patient-level split isolation and leakage/confounding caution. |
| Stage 9 two-stream ensemble recipe optimization | `dietterich2000ensemble`, `caruana2004ensemble`, `akiba2019optuna`, `bergstra2011algorithms`, `saito2015precision`, `matthews1975comparison` | Cite ensemble-method and ensemble-selection papers for combining trained model candidates, Optuna/TPE for semantic/spatial weight and ROI-threshold search, Saito/Rehmsmeier for AUPRC objectives under imbalance, and Matthews for MCC-based decision-threshold calibration. The two-stream ROI-gating objective is repository-specific. |
| Stage 9 prediction caching, TTA, validation split, and recipe provenance | `moshkov2020test`, `folk2011hdf5`, `wilkinson2016fair`, `piccolo2016tools`, `roberts2017crossvalidation`, `dawood2026confounding` | Cite Moshkov et al. for test-time augmentation in segmentation, HDF5/FAIR/Piccolo for cached prediction and recipe provenance, and Roberts/Dawood for patient-level optimization/calibration/holdout splitting. |
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
- Stage 5 sanity checks: cite Piccolo and Frampton (`piccolo2016tools`) and
  FAIR (`wilkinson2016fair`) for manifest/run-config/split-stats provenance and
  the fail-closed reproducibility gate. Cite Roberts et al.
  (`roberts2017crossvalidation`) and Dawood et al. (`dawood2026confounding`)
  for patient-level leakage checks, Folk et al. (`folk2011hdf5`) when
  describing HDF5-backed row parity and decodability, Yu et al.
  (`yu2020noisy`) for mask-label semantic integrity, and He/Garcia
  (`he2009imbalanced`) for reporting class imbalance across patches and
  patients.
- Stage 6 smart sampling: cite Campanella et al. (`campanella2025clinical`)
  for peer-reviewed context on public self-supervised pathology foundation
  models and Oquab et al. (`oquab2024dinov2`) for DINOv2-style feature
  extraction. Cite He/Garcia (`he2009imbalanced`) when explaining why positive
  label or mask-positive rows are protected before sampling the reducible pool.
- Stage 6 adaptive selection: cite Hubert and Arabie (`hubert1985comparing`) for
  the adjusted Rand index used in the stability loop, Sculley
  (`sculley2010webscale`) for MiniBatchKMeans, and Fahrbach et al.
  (`fahrbach2025gist`) only when `SMART_SAMPLER_USE_GIST=True`.
- Stage 6 outputs and compact TRAIN_SELECTED: cite HDF5 (`folk2011hdf5`) for
  compact HDF5 shards and FAIR/Piccolo (`wilkinson2016fair`,
  `piccolo2016tools`) for sidecars, row mappings, summary JSON, run
  configuration, and manifest-backed row-state updates.
- Stage 7 LR finder: cite Smith (`smith2017cyclical`) for the learning-rate
  range-test rationale, McKay et al. (`mckay1979comparison`) for
  Latin-hypercube sampling over the loss-weight space, Khened et al.
  (`khened2021generalized`) for the weighted BCE-plus-Dice loss being screened,
  and Loshchilov/Hutter (`loshchilov2019decoupled`) for AdamW.
- Stage 7 runtime controls: cite Micikevicius et al.
  (`micikevicius2018mixed`) only when discussing enabled mixed precision or AMP
  configuration; the current LR-finder default is `fp32`. Cite He/Garcia
  (`he2009imbalanced`) for the inverse-frequency `WeightedRandomSampler`, and
  cite FAIR/Piccolo (`wilkinson2016fair`, `piccolo2016tools`) for the LR-finder
  run config, environment capture, summary CSVs, plots, and LaTeX/PDF report.
- Stage 7 compact and normalization provenance: cite HDF5 (`folk2011hdf5`) for
  compact TRAIN_SELECTED storage when used, and reuse the Stage 4 normalization
  citations for the selected runtime-normalization method: Reinhard
  (`reinhard2001color`), Ruifrok (`ruifrok2001quantification`), Macenko
  (`macenko2009method`), Vahadane (`vahadane2016structure`), plus
  Tellez/Duenweg (`tellez2019augmentation`, `duenweg2023scanner`) for
  stain/scanner variability motivation.
- Stage 8 registered architectures: cite the architecture/backbone source for
  the selected registry pair. Current active registry pairs are U-Net plus Swin
  (`ronneberger2015unet`, `liu2021swin`), DeepLabv3+ plus ResNeSt
  (`chen2018deeplabv3plus`, `zhang2022resnest`), UNet++ plus EfficientNet
  (`zhou2020unetpp`, `tan2019efficientnet`), FPN plus SENet
  (`lin2017fpn`, `hu2018senet`), SegFormer/MiT (`xie2021segformer`),
  MA-Net plus ResNet (`fan2020manet`, `he2016resnet`), DPT plus ViT
  (`ranftl2021dpt`, `dosovitskiy2021vit`), and UPerNet plus Hiera
  (`xiao2018upernet`, `ryali2023hiera`). Cite ImageNet/ILSVRC
  (`russakovsky2015imagenet`) when explaining pretrained encoder weights or
  ImageNet normalization.
- Stage 8 training procedure: cite Khened et al. (`khened2021generalized`) for
  the BCE-plus-Dice loss, He/Garcia (`he2009imbalanced`) for inverse-frequency
  weighted sampling, AdamW (`loshchilov2019decoupled`) and Schedule-Free
  learning (`defazio2024road`) for the optimizer choices, Micikevicius et al.
  (`micikevicius2018mixed`) for AMP/mixed precision, Shrivastava et al.
  (`shrivastava2016ohem`) only when OHEM is enabled, and Prechelt
  (`prechelt1998automatic`) for validation-based early stopping.
- Stage 8 validation and checkpointing: cite Saito/Rehmsmeier
  (`saito2015precision`) for emphasizing AUPRC/precision-recall under class
  imbalance, Matthews (`matthews1975comparison`) for MCC, and FAIR/Piccolo
  (`wilkinson2016fair`, `piccolo2016tools`) for run metadata, compatibility
  signatures, runtime environment capture, checkpoint sidecars, and Aim logs.
- Stage 8 manifest and normalization provenance: cite Roberts/Dawood
  (`roberts2017crossvalidation`, `dawood2026confounding`) for patient-level
  train/validation isolation, HDF5 (`folk2011hdf5`) for canonical and compact
  row storage, and reuse the Stage 4 runtime-normalization citations for the
  selected on-the-fly normalization method.
- Stage 9 ensemble optimizer: cite Dietterich (`dietterich2000ensemble`) for
  the general ensemble-learning framing and Caruana et al.
  (`caruana2004ensemble`) for validation-set ensemble selection from a library
  of trained models. Cite Optuna (`akiba2019optuna`) and TPE
  (`bergstra2011algorithms`) for the semantic/spatial weight and ROI-threshold
  searches, while describing the two-stream semantic ROI gate plus spatial
  spill-penalized objective as repository-specific.
- Stage 9 TTA and validation metrics: cite Moshkov et al. (`moshkov2020test`)
  when describing horizontal/vertical flip test-time augmentation for
  segmentation probabilities. Cite Saito/Rehmsmeier (`saito2015precision`) for
  AUPRC under class imbalance and Matthews (`matthews1975comparison`) for the
  MCC criterion used to calibrate the final decision threshold.
- Stage 9 validation splitting and recipe provenance: cite Roberts/Dawood
  (`roberts2017crossvalidation`, `dawood2026confounding`) for keeping
  optimization, calibration, and holdout groups patient-disjoint within the
  validation set; cite HDF5 (`folk2011hdf5`) for cached prediction storage and
  FAIR/Piccolo (`wilkinson2016fair`, `piccolo2016tools`) for the recipe JSON,
  run config, compatibility signature, selected-model metadata, and validation
  provenance.
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
- Do not claim Stage 5 proves clinical validity or eliminates every possible
  hidden confounder. It validates declared repository contracts: manifests,
  run-configs, split statistics, patient isolation, HDF5 row parity, mask values,
  and mask-label semantics.
- Do not cite Piccolo and Frampton or FAIR as if they specify the repository's exact
  `manifest.csv`, `run_config.json`, `split_stats.csv`, HDF5 schema, or final
  verdict names. They support the reproducibility/provenance rationale only.
- Do not describe Stage 5 sampled path/shape/mask scans as exhaustive unless the
  corresponding full-scan flags were enabled for that run.
- Do not cite He and Garcia as prescribing the repository's patch-per-patient
  skew threshold. They support documenting class imbalance; the `max > 20x
  median` warning is a local heuristic.
- Do not cite the exact default model string `owkin/phikon-v2` as if the model
  card or arXiv preprint is a peer-reviewed method paper. For peer-reviewed
  manuscript support, cite Campanella et al. for the public pathology
  foundation-model benchmark and Oquab et al. for DINOv2-style self-supervised
  visual features; separately report the exact model identifier for
  reproducibility.
- Do not claim Stage 6 is a published smart-sampling algorithm. The protected
  retention policy, adaptive budget, holdout coverage plateau, and compact
  artifact flow are repository-specific decisions.
- Do not describe the large-pool GIST path as paper-exact GIST. The code first
  builds a bounded landmark pool with MiniBatchKMeans and then applies the
  GIST-style facility-location selector, so the full search space and theoretical
  guarantee from Fahrbach et al. no longer apply.
- Do not describe compact `TRAIN_SELECTED` storage as a new validation or test
  dataset. It is a TRAIN-only I/O artifact derived from Stage 6 selected rows;
  validation and test remain tied to canonical Phase 2 patient shards.
- Do not present Stage 7 outputs as final model performance, final model
  selection, or validation/test evidence. Stage 7 runs repeated training-range
  probes on TRAIN or TRAIN_SELECTED rows and records validation-split provenance,
  but the ranking criterion is median minimum smoothed training loss.
- Do not describe the Stage 7 exponential LR sweep as a full cyclical learning
  rate schedule. Smith supports the LR range-test idea; the repository's start
  LR, end LR, smoothing, divergence stop, and partial-history behavior are local
  implementation choices.
- Do not claim Latin-hypercube sampling optimizes the BCE/Dice weights. McKay et
  al. support stratified space-filling sampling; the selected bounds and
  number of samples are repository configuration.
- Do not cite mixed-precision training when Stage 7 is run with the default
  `LR_FINDER_AMP_PRECISION=fp32`, except to explain why AMP was available but
  disabled for stability.
- Do not imply compact TRAIN_SELECTED changes validation or test semantics in
  Stage 7. Compact storage can remap the training rows only; validation
  provenance remains canonical.
- Do not cite the architecture papers as evidence that a registry pair is
  superior on this pathology task. They support the model-family descriptions;
  performance claims must come from repository experiments.
- Do not describe each Stage 8 registry key as a full standalone paper
  implementation. For example, `SWIN` is built as an SMP U-Net decoder with a
  Swin encoder, and `DPT`, `SEGFORMER`, and `UPERNET` are instantiated through
  the library wrappers and configured encoders.
- Do not cite ImageNet/ILSVRC as pathology evidence. Use
  `russakovsky2015imagenet` only for pretrained weights, transfer-learning
  context, or ImageNet normalization assumptions.
- Do not present Stage 8 validation metrics as final held-out TEST evidence.
  Stage 8 uses VALIDATION AUPRC for best-checkpoint selection and early
  stopping; final test/inference reporting belongs downstream.
- Do not cite Schedule-Free learning unless `TRAINING_OPTIMIZER_NAME` is
  `AdamWScheduleFree`, and do not claim the repository reproduces all
  benchmark settings from Defazio et al. Cite AdamW separately for decoupled
  weight decay.
- Do not cite OHEM unless `TRAINING_RUN_OHEM=True`. Shrivastava et al. support
  the online hard-example-mining idea; this repository applies a pixel-level
  segmentation-loss variant, not the original region-based detector pipeline.
- Do not present Stage 9 optimization/calibration/holdout metrics as final TEST
  evidence. Stage 9 partitions VALIDATION patients to produce an ensemble recipe
  for Stage 10; the held-out TEST evaluation remains downstream.
- Do not cite Caruana et al. as if Stage 9 implements their exact greedy
  forward ensemble-selection algorithm. Stage 9 performs library-based model
  selection and Optuna-optimized continuous weights, so the citation supports
  the ensemble-selection motivation rather than a paper-exact algorithm.
- Do not cite the ROI gate or spill penalty as a published method. The semantic
  low-pass ROI mask, empty/permissive ROI pruning, and spatial spill penalty are
  repository-specific design choices.
- Do not describe the decision threshold as probability calibration. Stage 9
  sweeps thresholds from 0.05 to 0.95 and selects the value with the best mean
  patient MCC on calibration patients.
- Do not cite test-time augmentation unless the flip-averaged prediction path is
  discussed. Stage 9 averages the original, horizontal-flip, and vertical-flip
  probabilities; it does not perform a larger TTA policy search.
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

## Stage 5 Sanity Checks Notes

Suggested code boundary: `5_sanity_checks.py`, `helpers/sanity/config.py`,
`helpers/sanity/pipeline.py`, `helpers/sanity/manifest_checks.py`,
`helpers/sanity/contracts.py`, `helpers/sanity/disk_checks.py`,
`helpers/sanity/semantic_checks.py`, `helpers/sanity/provenance.py`, and
`helpers/sanity/reporting.py`.

| Stage 5 decision | Code behavior to document | Suggested citation |
| --- | --- | --- |
| Fail-closed scientific QA gate | Stage 5 loads `manifest.csv`, `run_config.json`, and `split_stats.csv`, runs dataset-wide and split-level checks, prints a scientific integrity report, returns `SPLITS PASSED` only when no check has status `FAIL`, and exits with code 2 on failed verdicts or fatal errors. | `piccolo2016tools`; `wilkinson2016fair` |
| Manifest, run-config, and split-stats contracts | The stage validates required manifest columns, split names, labels, nonnegative source rows, duplicate rows, canonical `source_hdf5_path`/`source_row_index`, paired logical HDF5 image/mask references, run-config patient lists and constraints, recomputed split-stat parity, and Stage 3.3 cleaning lineage. | `wilkinson2016fair`; `piccolo2016tools` |
| Patient-level leakage check | Stage 5 fails if a `patient_id` appears in more than one of TRAIN, VALIDATION, and TEST, and cross-checks the manifest patient sets against `run_config.json` when available. | `roberts2017crossvalidation`; `dawood2026confounding` |
| HDF5 row parity and decodability | For each split, Stage 5 verifies referenced HDF5 files, row bounds, filename/label/patient-id parity between manifest and HDF5 datasets, image/mask shape agreement, and decodability of sampled or full-scan rows. | `folk2011hdf5`; `wilkinson2016fair` |
| Binary mask and label semantics | Stage 5 checks that masks contain only binary values, that cancer-labeled rows have positive mask pixels, and that not-cancer-labeled rows do not contain positive mask pixels, with fail/warn behavior controlled by sanity configuration. | `yu2020noisy`; exact binary semantics are repository-specific |
| Class and patient-distribution visibility | Stage 5 reports patch-level class counts, patient-level class counts, split class presence, and patches-per-patient summaries, warning when patch counts are extremely skewed by patient. | `he2009imbalanced`; `roberts2017crossvalidation`; the `max > 20x median` threshold is repository-specific |
| Sampled versus full-scan integrity policy | The default path samples path/shape/mask checks with fixed seeds and exposes `SANITY_FULL_MASK_SCAN` and `SANITY_FULL_SHAPE_SCAN` when exhaustive scanning is required. | `piccolo2016tools` for reproducibility reporting; the sampling policy itself is repository-specific |

## Stage 6 Smart Sampler Notes

Suggested code boundary: `6_smart_sampler.py`, `helpers/smart_sampling/config.py`,
`helpers/smart_sampling/pipeline.py`, `helpers/smart_sampling/index.py`,
`helpers/smart_sampling/embeddings.py`, `helpers/smart_sampling/selection.py`,
`helpers/smart_sampling/gist.py`, `helpers/smart_sampling/storage.py`,
`helpers/smart_sampling/writer.py`, and
`helpers/training/compact_train_selected.py`.

| Stage 6 decision | Code behavior to document | Suggested citation |
| --- | --- | --- |
| Manifest-backed TRAIN-only source contract | Stage 6 builds its patient index from `master_manifest.sqlite`, using rows where `split = TRAIN` and `is_stage4_accepted = 1`, and requires each patient to map to one canonical source HDF5 shard. | `wilkinson2016fair`; optionally `roberts2017crossvalidation` when explaining patient-wise structure |
| Pathology foundation-model embeddings | The default extractor loads `owkin/phikon-v2` through Hugging Face, runs CUDA inference, and uses the first-token descriptor from `last_hidden_state` as the embedding for patch selection. | `campanella2025clinical`; `oquab2024dinov2`; report the exact model identifier separately because the Phikon-v2 source paper/model card is not the peer-reviewed anchor here |
| Protected retention before negative/reducible sampling | Rows with positive labels and, by default, any mask-positive rows are retained before the reducer samples the remaining patient-specific pool. | `he2009imbalanced`; optionally `yu2020noisy` when framing this as label/mask integrity protection |
| Adaptive embedding budget and stability stop | For each patient, Stage 6 starts with a configurable subset size, repeatedly samples two subsets, clusters both with MiniBatchKMeans, scores agreement with adjusted Rand index, and grows the subset until the stability threshold, maximum count, or step cap is reached. | `hubert1985comparing`; `sculley2010webscale`; thresholds and growth policy are repository-specific |
| Default cluster-balanced diversity selector | The default reducer clusters embeddings with MiniBatchKMeans, builds a cluster-balanced candidate order, optionally evaluates a within-patient holdout by mean nearest-selected embedding distance, and stops when coverage improvement plateaus. | `sculley2010webscale`; the quota/order/plateau policy is repository-specific |
| Optional GIST-style selector | When `SMART_SAMPLER_USE_GIST=True`, the reducer selects a diverse subset with a facility-location plus diversity objective over embeddings. If the candidate pool exceeds the configured limit, Stage 6 first constructs a MiniBatchKMeans landmark pool and then applies the GIST-style selector. | `fahrbach2025gist`; call the large-pool mode GIST-style or GIST-inspired, not paper-exact GIST |
| Sidecars before manifest updates | Stage 6 writes `train_filtered_selection.csv`, `patient_filter_stats.csv`, `filter_run_config.json`, and `filter_summary.json`, then records the run and writes `sampling_decision` / `is_stage7_selected` row states to `master_manifest.sqlite`. | `wilkinson2016fair`; `piccolo2016tools` |
| Compact TRAIN_SELECTED artifact | When enabled, Stage 6 builds compact HDF5 shards containing only selected TRAIN rows, validates filename/label/patient parity against the canonical source rows, writes `index.sqlite` and `summary.json`, and publishes the completed artifact for Stage 7/8 TRAIN reads. | `folk2011hdf5`; `wilkinson2016fair` |
| Disk-constrained staging controls | Input staging, sidecar staging, local cache size, compact-build scratch path, and cleanup flags are operational controls for storage pressure. They should be reported for reproducibility when used, but they are not independent scientific-method citations. | `piccolo2016tools` for reproducibility reporting only |

## Stage 7 LR Finder Notes

Suggested code boundary: `7_lr_finder.py`, `helpers/lr_finder/config.py`,
`helpers/lr_finder/search_space.py`, `helpers/lr_finder/data.py`,
`helpers/lr_finder/runner.py`, `helpers/lr_finder/analysis.py`,
`helpers/lr_finder/pipeline.py`, `helpers/lr_finder/reporting.py`,
`helpers/training/losses.py`, `helpers/training/compact_train_selected.py`,
and `helpers/training/stain_normalization.py`.

| Stage 7 decision | Code behavior to document | Suggested citation |
| --- | --- | --- |
| Registry-bounded architecture screening | Stage 7 builds model plans from the approved training-model registry and can filter architectures, but it does not open arbitrary model choices at runtime. | Cite architecture papers separately only when describing the actual model families; registry gating itself is repository-specific and mainly a reproducibility control (`piccolo2016tools`). |
| Latin-hypercube loss-weight search | The stage samples `alpha`, `beta`, and `gamma` for the BCE/background-Dice/foreground-Dice hybrid loss with a fixed-seed Latin-hypercube design over configured bounds. | `mckay1979comparison`; `khened2021generalized` |
| Direct exponential LR range test | For each architecture, encoder, loss-weight sample, and repeat, Stage 7 initializes AdamW at `LR_FINDER_OPTIMIZER_START_LR`, exponentially increases LR to `LR_FINDER_END_LR`, smooths the loss, stops on divergence or non-finite loss, and records partial history. | `smith2017cyclical`; `loshchilov2019decoupled`; the exact smoothing and divergence rules are repository-specific |
| Repeated screening and ranking criterion | The same sampled loss configuration is repeated with deterministic seed offsets, and valid repeats are summarized by the median of the minimum smoothed loss after trimming early/late curve points. | `smith2017cyclical`; repeated-median ranking is a local screening heuristic, not final validation evidence |
| Pretrained-state reuse across loss configurations | For each architecture/encoder pair, Stage 7 loads the pretrained model once, snapshots its initial state to CPU, and reloads that same state for every loss sample and repeat. | `piccolo2016tools` for reproducible screening discipline; exact state-snapshot implementation is repository-specific |
| Weighted sampling under class imbalance | Stage 7 computes inverse class-frequency sample weights from the selected training dataset and uses PyTorch `WeightedRandomSampler` with replacement for the LR-range batches. | `he2009imbalanced`; sampler implementation is library-specific |
| Smart-sampling and compact TRAIN_SELECTED consumption | If smart sampling is enabled, Stage 7 queries only Stage 6 selected TRAIN rows; if compact storage is enabled, it remaps those rows to compact TRAIN_SELECTED HDF5 shards while keeping validation provenance canonical. | `folk2011hdf5`; `wilkinson2016fair`; see Stage 6 notes for compact artifact generation |
| Runtime stain normalization during screening | Stage 7 constructs the dataset with the selected runtime-normalization method and records the active method/backend in provenance; MACENKO/VAHADANE may still estimate per-patch source stain information at runtime. | `tellez2019augmentation`; `duenweg2023scanner`; method-specific citations `reinhard2001color`, `ruifrok2001quantification`, `macenko2009method`, `vahadane2016structure` |
| Precision and CUDA-OOM controls | The LR finder supports AMP precision choices, defaults to `fp32`, retries CUDA OOM repeats with smaller effective batch sizes, and reports the effective batch size. | `micikevicius2018mixed` only if AMP is enabled/discussed; OOM retry is repository-specific engineering |
| Reproducibility outputs and reporting | Stage 7 writes `LHS_SAMPLES.json`, `SUMMARY_ALL.csv`, per-architecture summaries, LR/loss plots, `lr_finder_run_config.json`, runtime environment metadata, and a LaTeX/PDF report. | `wilkinson2016fair`; `piccolo2016tools` |

## Stage 8 Training Ensemble Notes

Suggested code boundary: `8_training_ensemble.py`, `helpers/training/config.py`,
`helpers/training/registry.py`, `helpers/training/models.py`,
`helpers/training/data.py`, `helpers/training/loop.py`,
`helpers/training/losses.py`, `helpers/training/pipeline.py`,
`helpers/training/checkpointing.py`, `helpers/training/metrics.py`,
`helpers/training/runtime.py`, `helpers/training/gpu.py`,
`helpers/training/reporting.py`, `helpers/training/compact_train_selected.py`,
and the active `training_model_registry_*.json` files.

Active architecture and encoder papers:

| Registry key | Current SMP/model behavior and approved encoder | Suggested citation |
| --- | --- | --- |
| `SWIN` | Builds an SMP `Unet` decoder with `tu-swin_large_patch4_window7_224.ms_in22k_ft_in1k`. | `ronneberger2015unet`; `liu2021swin`; `russakovsky2015imagenet` for pretrained weights |
| `DEEPLABV3PLUS` | Builds SMP `DeepLabV3Plus` with `tu-resnest101e`. | `chen2018deeplabv3plus`; `zhang2022resnest`; `russakovsky2015imagenet` |
| `UNET++` | Builds SMP `UnetPlusPlus` with `efficientnet-b7`. | `zhou2020unetpp`; `tan2019efficientnet`; `russakovsky2015imagenet` |
| `FPN` | Builds SMP `FPN` with `senet154`. | `lin2017fpn`; `hu2018senet`; `russakovsky2015imagenet` |
| `SEGFORMER` | Builds SMP `Segformer` with `mit_b5`. | `xie2021segformer`; `russakovsky2015imagenet` when discussing pretrained weights |
| `MANET` | Builds SMP `MAnet` with `resnet152`. | `fan2020manet`; `he2016resnet`; `russakovsky2015imagenet` |
| `DPT` | Builds SMP `DPT` with `tu-vit_large_patch16_224.augreg_in21k_ft_in1k`. | `ranftl2021dpt`; `dosovitskiy2021vit`; `russakovsky2015imagenet` |
| `UPERNET` | Builds SMP `UPerNet` with `tu-hiera_large_224`. | `xiao2018upernet`; `ryali2023hiera`; `russakovsky2015imagenet` |

Training and provenance decisions:

| Stage 8 decision | Code behavior to document | Suggested citation |
| --- | --- | --- |
| Registry-gated architecture/encoder selection | `TRAINING_ARCHITECTURE` and `TRAINING_ENCODER` must match the configured registry; the checked-in method-specific registries share the same eight architecture keys and approved encoders while varying LR by normalization method. | Architecture citations above; `piccolo2016tools` for reproducible configuration discipline |
| ImageNet-pretrained encoder initialization | `create_model()` uses ImageNet/pretrained weights for non-validation runs, with `encoder_weights=True` for `timm` encoders and `"imagenet"` for SMP encoders. | `russakovsky2015imagenet`; architecture/backbone citations for the selected pair |
| Manifest-backed TRAIN/VALIDATION datasets | Stage 8 queries `master_manifest.sqlite` once, loads TRAIN rows or Stage 6 selected TRAIN rows, loads canonical VALIDATION rows, prefetches patient shards to local fast storage, and checks train/validation patient separation before training. | `wilkinson2016fair`; `roberts2017crossvalidation`; `dawood2026confounding`; `folk2011hdf5` |
| Compact TRAIN_SELECTED consumption | When smart sampling and compact storage are enabled, TRAIN rows are remapped to the compact Stage 6 HDF5 artifact; VALIDATION remains canonical. | `folk2011hdf5`; `wilkinson2016fair`; see Stage 6 compact-artifact notes |
| Runtime stain normalization during training | The train and validation datasets are constructed with the selected runtime-normalization method/backend and record normalization artifact lineage in metadata. | `tellez2019augmentation`; `duenweg2023scanner`; method-specific citations `reinhard2001color`, `ruifrok2001quantification`, `macenko2009method`, `vahadane2016structure` |
| Weighted sampling under class imbalance | Stage 8 computes inverse class-frequency sample weights and uses a seeded PyTorch `WeightedRandomSampler` with replacement for training. | `he2009imbalanced`; sampler implementation is repository/library-specific |
| Training augmentation and ImageNet normalization | Stage 8 applies Albumentations flips, affine transforms, color/noise/blur, coarse dropout, then GPU ImageNet mean/std normalization and stochastic downscale/upscale. | `buslaev2020albumentations`; `dossantos2023augmentation`; `tellez2019augmentation`; `shorten2019survey`; `zhong2020random`; `devries2017cutout`; `russakovsky2015imagenet` |
| BCE-plus-Dice segmentation loss | Stage 8 uses `BCEDiceHybridLossPaper`, a direct implementation of Khened et al.'s weighted BCE/background-Dice/foreground-Dice family with registry-provided weights. | `khened2021generalized` |
| Optional pixel-level OHEM | If enabled, Stage 8 starts OHEM after the configured epoch, ranks pixels by BCE error, keeps the hardest pixels subject to ratio and minimum-count controls, and disables OHEM during validation. | `shrivastava2016ohem`; repository implementation is pixel-level segmentation OHEM, not paper-exact detector OHEM |
| Optional artifact-aware sample discount | If enabled, Stage 8 appends filename-keyed artifact coverage covariates from the manifest and discounts per-sample loss by maximum artifact coverage. | `weng2024grandqc` for artifact/QC motivation; exact discount rule is repository-specific |
| Optimizer choice and weight decay | Stage 8 supports `AdamW` and `AdamWScheduleFree`; LR and WD come from the selected registry entry and are recorded in Aim and metadata. | `loshchilov2019decoupled`; `defazio2024road` when `AdamWScheduleFree` is used |
| AMP and accumulation controls | Stage 8 resolves `fp16`, `bf16`, `fp32`, or `auto` precision, forces DPT to `fp32`, uses gradient scaling for `fp16`, and supports gradient accumulation. | `micikevicius2018mixed`; exact architecture-specific precision rules are repository-specific |
| Validation metrics and best checkpoint | Validation accumulates pixel histograms for AUPRC, AUROC, and MCC*, guards against collapsed foreground prevalence, and saves the best checkpoint by validation AUPRC with early stopping. | `saito2015precision`; `matthews1975comparison`; `prechelt1998automatic`; `sokolova2009performance` |
| Checkpoint resume compatibility and metadata | Resume requires matching provenance signatures for split, packaging, normalization, smart-sampling, artifact-aware loss, and OHEM settings; final metadata captures runtime environment, dataset lineage, optimizer/loss settings, and metrics. | `wilkinson2016fair`; `piccolo2016tools` |
| Aim logging and completion reporting | Stage 8 logs hparams and epoch metrics to Aim when available and writes completion metadata/email summaries. | `piccolo2016tools`; `wilkinson2016fair` |

## Stage 9 Ensemble Optimizer Notes

Suggested code boundary: `9_optimizer_ensemble.py`,
`helpers/ensemble_optimizer/config.py`, `helpers/ensemble_optimizer/pipeline.py`,
`helpers/ensemble_optimizer/metadata.py`, `helpers/ensemble_optimizer/models.py`,
`helpers/ensemble_optimizer/data.py`, `helpers/ensemble_optimizer/splitting.py`,
`helpers/ensemble_optimizer/optimization.py`,
`helpers/ensemble_optimizer/reporting.py`, `helpers/provenance.py`, and
the Stage 8 model metadata sidecars consumed from `ENSEMBLE_OPT_METADATA_DIR`.

| Stage 9 decision | Code behavior to document | Suggested citation |
| --- | --- | --- |
| Metadata-driven model-library selection | Stage 9 loads `*_meta.json` files, requires compatible fail-closed provenance signatures, ranks candidates on the optimization subset, and selects the best valid model per requested architecture. | `caruana2004ensemble`; `dietterich2000ensemble`; `wilkinson2016fair`; `piccolo2016tools` |
| Semantic and spatial architecture streams | The default semantic stream is `SWIN,DPT,SEGFORMER,UPERNET`; the default spatial stream is `DEEPLABV3PLUS,UNET++,FPN,MANET`. The stream split is a repository design over the Stage 8 registry families. | Cite the Stage 8 architecture papers for each selected model; stream assignment itself is repository-specific |
| Manifest-backed VALIDATION source contract | Stage 9 resolves canonical VALIDATION rows from `master_manifest.sqlite`, optionally stages patient shards locally, applies the selected runtime normalization, and records validation provenance. | `wilkinson2016fair`; `folk2011hdf5`; `roberts2017crossvalidation`; `dawood2026confounding`; normalization citations as in Stage 4 |
| Patient-level optimization/calibration/holdout split | Validation patients are split into disjoint optimization, calibration, and optional development holdout groups with a fixed seed and positive/negative patient awareness. | `roberts2017crossvalidation`; `dawood2026confounding`; exact split-count policy is repository-specific |
| Batched TTA prediction cache | Each selected model is evaluated on validation patches with original, horizontal-flip, and vertical-flip predictions averaged; foreground probabilities are cached as uint16 memmaps with truth masks and patient ids. | `moshkov2020test`; `folk2011hdf5`; cache quantization and storage layout are repository-specific |
| Optuna/TPE semantic stream optimization | The semantic stream optimizes normalized model weights and an ROI threshold with `TPESampler`; trials are pruned if ROIs are too empty, miss positive tissue, or become too permissive. | `akiba2019optuna`; `bergstra2011algorithms`; ROI constraints are repository-specific |
| Low-pass ROI gate | The semantic probability ensemble is downsampled, thresholded, and upsampled to form a coarse ROI mask that gates the spatial stream. | Cite `akiba2019optuna` only for the threshold search machinery; the low-pass ROI gate is repository-specific |
| Optuna/TPE spatial stream optimization | With the semantic ROI fixed, Stage 9 optimizes spatial model weights for a composite objective: macro positive-patient AUPRC inside ROI minus spill and negative false-positive penalties. | `akiba2019optuna`; `bergstra2011algorithms`; `saito2015precision`; penalty terms are repository-specific |
| Decision threshold calibration | Stage 9 sweeps thresholds from 0.05 to 0.95 on calibration patients and selects the threshold maximizing mean patient MCC after ROI gating. | `matthews1975comparison`; threshold sweep granularity is repository-specific |
| Development holdout recipe check | The optimized semantic weights, spatial weights, ROI threshold, and calibrated decision threshold are evaluated on the holdout validation patients with macro AUPRC, spill, negative-FP, and composite objective summaries. | `saito2015precision`; `matthews1975comparison`; holdout is validation-internal, not final TEST evidence |
| Recipe and run-config provenance | Stage 9 writes `ENSEMBLE_TWO_STREAM_*.json` and `ensemble_optimizer_run_config.json` with selected model metadata, checkpoint hashes, stream weights, thresholds, calibration metrics, holdout metrics, validation lineage, split fingerprint, and recipe signature. | `wilkinson2016fair`; `piccolo2016tools` |

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

@inproceedings{smith2017cyclical,
  title = {Cyclical Learning Rates for Training Neural Networks},
  author = {Smith, Leslie N.},
  booktitle = {2017 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages = {464--472},
  year = {2017},
  doi = {10.1109/WACV.2017.58},
  url = {https://doi.org/10.1109/WACV.2017.58}
}

@article{mckay1979comparison,
  title = {A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code},
  author = {McKay, M. D. and Beckman, R. J. and Conover, W. J.},
  journal = {Technometrics},
  volume = {21},
  number = {2},
  pages = {239--245},
  year = {1979},
  doi = {10.1080/00401706.1979.10489755},
  url = {https://doi.org/10.1080/00401706.1979.10489755}
}

@inproceedings{loshchilov2019decoupled,
  title = {Decoupled Weight Decay Regularization},
  author = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {International Conference on Learning Representations},
  year = {2019},
  url = {https://openreview.net/forum?id=Bkg6RiCqY7}
}

@inproceedings{micikevicius2018mixed,
  title = {Mixed Precision Training},
  author = {Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and Wu, Hao},
  booktitle = {International Conference on Learning Representations},
  year = {2018},
  url = {https://openreview.net/forum?id=r1gs9JgRZ}
}

@inproceedings{ronneberger2015unet,
  title = {{U-Net}: Convolutional Networks for Biomedical Image Segmentation},
  author = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2015},
  series = {Lecture Notes in Computer Science},
  volume = {9351},
  pages = {234--241},
  year = {2015},
  publisher = {Springer},
  doi = {10.1007/978-3-319-24574-4_28},
  url = {https://doi.org/10.1007/978-3-319-24574-4_28}
}

@inproceedings{liu2021swin,
  title = {Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows},
  author = {Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages = {10012--10022},
  year = {2021},
  doi = {10.1109/ICCV48922.2021.00986},
  url = {https://doi.org/10.1109/ICCV48922.2021.00986}
}

@inproceedings{chen2018deeplabv3plus,
  title = {Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author = {Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  pages = {801--818},
  year = {2018},
  url = {https://openaccess.thecvf.com/content_ECCV_2018/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.html}
}

@inproceedings{zhang2022resnest,
  title = {{ResNeSt}: Split-Attention Networks},
  author = {Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Lin, Haibin and Zhang, Zhi and Sun, Yue and He, Tong and Mueller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages = {2735--2745},
  year = {2022},
  doi = {10.1109/CVPRW56347.2022.00309},
  url = {https://doi.org/10.1109/CVPRW56347.2022.00309}
}

@article{zhou2020unetpp,
  title = {{UNet++}: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation},
  author = {Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
  journal = {IEEE Transactions on Medical Imaging},
  volume = {39},
  number = {6},
  pages = {1856--1867},
  year = {2020},
  doi = {10.1109/TMI.2019.2959609},
  url = {https://doi.org/10.1109/TMI.2019.2959609}
}

@inproceedings{tan2019efficientnet,
  title = {{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks},
  author = {Tan, Mingxing and Le, Quoc V.},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {97},
  pages = {6105--6114},
  year = {2019},
  url = {https://proceedings.mlr.press/v97/tan19a.html}
}

@inproceedings{lin2017fpn,
  title = {Feature Pyramid Networks for Object Detection},
  author = {Lin, Tsung-Yi and Doll{\'a}r, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {936--944},
  year = {2017},
  doi = {10.1109/CVPR.2017.106},
  url = {https://doi.org/10.1109/CVPR.2017.106}
}

@inproceedings{hu2018senet,
  title = {Squeeze-and-Excitation Networks},
  author = {Hu, Jie and Shen, Li and Sun, Gang},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {7132--7141},
  year = {2018},
  doi = {10.1109/CVPR.2018.00745},
  url = {https://doi.org/10.1109/CVPR.2018.00745}
}

@inproceedings{xie2021segformer,
  title = {{SegFormer}: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author = {Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M. and Luo, Ping},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34},
  pages = {12077--12090},
  year = {2021},
  url = {https://papers.nips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html}
}

@article{fan2020manet,
  title = {{MA-Net}: A Multi-Scale Attention Network for Liver and Tumor Segmentation},
  author = {Fan, Tongle and Wang, Guanglei and Li, Yan and Wang, Hongrui},
  journal = {IEEE Access},
  volume = {8},
  pages = {179656--179665},
  year = {2020},
  doi = {10.1109/ACCESS.2020.3025372},
  url = {https://doi.org/10.1109/ACCESS.2020.3025372}
}

@inproceedings{he2016resnet,
  title = {Deep Residual Learning for Image Recognition},
  author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {770--778},
  year = {2016},
  doi = {10.1109/CVPR.2016.90},
  url = {https://doi.org/10.1109/CVPR.2016.90}
}

@inproceedings{ranftl2021dpt,
  title = {Vision Transformers for Dense Prediction},
  author = {Ranftl, Ren{\'e} and Bochkovskiy, Alexey and Koltun, Vladlen},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages = {12179--12188},
  year = {2021},
  url = {https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html}
}

@inproceedings{dosovitskiy2021vit,
  title = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle = {International Conference on Learning Representations},
  year = {2021},
  url = {https://openreview.net/forum?id=YicbFdNTTy}
}

@inproceedings{xiao2018upernet,
  title = {Unified Perceptual Parsing for Scene Understanding},
  author = {Xiao, Tete and Liu, Yingcheng and Zhou, Bolei and Jiang, Yuning and Sun, Jian},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  pages = {418--434},
  year = {2018},
  url = {https://openaccess.thecvf.com/content_ECCV_2018/html/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.html}
}

@inproceedings{ryali2023hiera,
  title = {Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles},
  author = {Ryali, Chaitanya and Hu, Yuan-Ting and Bolya, Daniel and Wei, Chen and Fan, Haoqi and Huang, Po-Yao and Aggarwal, Vaibhav and Chowdhury, Arkabandhu and Poursaeed, Omid and Hoffman, Judy and Malik, Jitendra and Li, Yanghao and Feichtenhofer, Christoph},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {202},
  pages = {29441--29454},
  year = {2023},
  url = {https://proceedings.mlr.press/v202/ryali23a.html}
}

@article{russakovsky2015imagenet,
  title = {{ImageNet} Large Scale Visual Recognition Challenge},
  author = {Russakovsky, Olga and Deng, Jia and Su, Hao and Krause, Jonathan and Satheesh, Sanjeev and Ma, Sean and Huang, Zhiheng and Karpathy, Andrej and Khosla, Aditya and Bernstein, Michael and Berg, Alexander C. and Fei-Fei, Li},
  journal = {International Journal of Computer Vision},
  volume = {115},
  number = {3},
  pages = {211--252},
  year = {2015},
  doi = {10.1007/s11263-015-0816-y},
  url = {https://doi.org/10.1007/s11263-015-0816-y}
}

@inproceedings{defazio2024road,
  title = {The Road Less Scheduled},
  author = {Defazio, Aaron and Yang, Xingyu and Mehta, Harsh and Mishchenko, Konstantin and Khaled, Ahmed and Cutkosky, Ashok},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  year = {2024},
  doi = {10.52202/079017-0320},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html}
}

@inproceedings{shrivastava2016ohem,
  title = {Training Region-Based Object Detectors with Online Hard Example Mining},
  author = {Shrivastava, Abhinav and Gupta, Abhinav and Girshick, Ross},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {761--769},
  year = {2016},
  doi = {10.1109/CVPR.2016.89},
  url = {https://doi.org/10.1109/CVPR.2016.89}
}

@article{prechelt1998automatic,
  title = {Automatic Early Stopping Using Cross Validation: Quantifying the Criteria},
  author = {Prechelt, Lutz},
  journal = {Neural Networks},
  volume = {11},
  number = {4},
  pages = {761--767},
  year = {1998},
  doi = {10.1016/S0893-6080(98)00010-0},
  url = {https://doi.org/10.1016/S0893-6080(98)00010-0}
}

@article{saito2015precision,
  title = {The Precision-Recall Plot Is More Informative than the {ROC} Plot When Evaluating Binary Classifiers on Imbalanced Datasets},
  author = {Saito, Takaya and Rehmsmeier, Marc},
  journal = {PLOS ONE},
  volume = {10},
  number = {3},
  pages = {e0118432},
  year = {2015},
  doi = {10.1371/journal.pone.0118432},
  url = {https://doi.org/10.1371/journal.pone.0118432}
}

@article{matthews1975comparison,
  title = {Comparison of the Predicted and Observed Secondary Structure of {T4} Phage Lysozyme},
  author = {Matthews, Brian W.},
  journal = {Biochimica et Biophysica Acta (BBA) - Protein Structure},
  volume = {405},
  number = {2},
  pages = {442--451},
  year = {1975},
  doi = {10.1016/0005-2795(75)90109-9},
  url = {https://doi.org/10.1016/0005-2795(75)90109-9}
}

@inproceedings{dietterich2000ensemble,
  title = {Ensemble Methods in Machine Learning},
  author = {Dietterich, Thomas G.},
  booktitle = {Multiple Classifier Systems},
  series = {Lecture Notes in Computer Science},
  volume = {1857},
  pages = {1--15},
  year = {2000},
  publisher = {Springer},
  doi = {10.1007/3-540-45014-9_1},
  url = {https://doi.org/10.1007/3-540-45014-9_1}
}

@inproceedings{caruana2004ensemble,
  title = {Ensemble Selection from Libraries of Models},
  author = {Caruana, Rich and Niculescu-Mizil, Alexandru and Crew, Geoff and Ksikes, Alex},
  booktitle = {Proceedings of the Twenty-First International Conference on Machine Learning},
  pages = {18},
  year = {2004},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/1015330.1015432},
  url = {https://doi.org/10.1145/1015330.1015432}
}

@article{moshkov2020test,
  title = {Test-Time Augmentation for Deep Learning-Based Cell Segmentation on Microscopy Images},
  author = {Moshkov, Nikita and Mathe, Botond and Kertesz-Farkas, Attila and Hollandi, Reka and Horvath, Peter},
  journal = {Scientific Reports},
  volume = {10},
  number = {1},
  pages = {5068},
  year = {2020},
  doi = {10.1038/s41598-020-61808-3},
  url = {https://doi.org/10.1038/s41598-020-61808-3}
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

@article{piccolo2016tools,
  title = {Tools and Techniques for Computational Reproducibility},
  author = {Piccolo, Stephen R. and Frampton, Michael B.},
  journal = {GigaScience},
  volume = {5},
  number = {1},
  pages = {30},
  year = {2016},
  doi = {10.1186/s13742-016-0135-4},
  url = {https://doi.org/10.1186/s13742-016-0135-4}
}

@inproceedings{folk2011hdf5,
  title = {An Overview of the HDF5 Technology Suite and Its Applications},
  author = {Folk, Mike and Heber, Gerd and Koziol, Quincey and Pourmal, Elena and Robinson, Dana},
  booktitle = {Proceedings of the 2011 EDBT/ICDT Workshop on Array Databases},
  series = {AD '11},
  pages = {36--47},
  year = {2011},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/1966895.1966900},
  url = {https://doi.org/10.1145/1966895.1966900}
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

@article{he2009imbalanced,
  title = {Learning from Imbalanced Data},
  author = {He, Haibo and Garcia, Edwardo A.},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume = {21},
  number = {9},
  pages = {1263--1284},
  year = {2009},
  doi = {10.1109/TKDE.2008.239},
  url = {https://doi.org/10.1109/TKDE.2008.239}
}

@article{campanella2025clinical,
  title = {A Clinical Benchmark of Public Self-Supervised Pathology Foundation Models},
  author = {Campanella, Gabriele and Chen, Shengjia and Singh, Manbir and others},
  journal = {Nature Communications},
  volume = {16},
  pages = {3640},
  year = {2025},
  doi = {10.1038/s41467-025-58796-1},
  url = {https://doi.org/10.1038/s41467-025-58796-1}
}

@article{oquab2024dinov2,
  title = {DINOv2: Learning Robust Visual Features without Supervision},
  author = {Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Assran, Mahmoud and Ballas, Nicolas and Galuba, Wojciech and Howes, Russell and Huang, Po-Yao and Li, Shang-Wen and Misra, Ishan and Rabbat, Michael and Sharma, Vasu and Synnaeve, Gabriel and Xu, Hu and J{\'e}gou, Herv{\'e} and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal = {Transactions on Machine Learning Research},
  year = {2024},
  url = {https://openreview.net/forum?id=a68SUt6zFt}
}

@article{hubert1985comparing,
  title = {Comparing Partitions},
  author = {Hubert, Lawrence and Arabie, Phipps},
  journal = {Journal of Classification},
  volume = {2},
  number = {1},
  pages = {193--218},
  year = {1985},
  doi = {10.1007/BF01908075},
  url = {https://doi.org/10.1007/BF01908075}
}

@inproceedings{sculley2010webscale,
  title = {Web-Scale K-Means Clustering},
  author = {Sculley, D.},
  booktitle = {Proceedings of the 19th International Conference on World Wide Web},
  pages = {1177--1178},
  year = {2010},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/1772690.1772862},
  url = {https://doi.org/10.1145/1772690.1772862}
}

@inproceedings{fahrbach2025gist,
  title = {GIST: Greedy Independent Set Thresholding for Max-Min Diversification with Submodular Utility},
  author = {Fahrbach, Matthew and Ramalingam, Srikumar and Zadimoghaddam, Morteza and Ahmadian, Sara and Citovsky, Gui and DeSalvo, Giulia},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2025},
  url = {https://research.google/pubs/gist-greedy-independent-set-thresholding-for-max-min-diversification-with-submodular-utility/}
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
