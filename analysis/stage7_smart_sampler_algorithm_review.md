# Stage 7 Smart Sampler: Algorithm Explanation and Scientific Validation Review

## Scope

This document explains the real logic behind `7_smart_sampler.py` and its helper modules:

- `7_smart_sampler.py`
- `helpers/smart_sampling/config.py`
- `helpers/smart_sampling/pipeline.py`
- `helpers/smart_sampling/selection.py`
- `helpers/smart_sampling/embeddings.py`
- `helpers/smart_sampling/index.py`
- `helpers/smart_sampling/writer.py`

The goal is to describe the algorithm precisely enough that another agent could re-implement it from this document alone, while also reviewing the method for scientific validity.

## Important Naming Note

The file is named `7_smart_sampler.py`, but its logs and status messages say `Stage 8 smart sampling completed successfully`. The helper pipeline also logs `Stage 7` in some places. Treat this as a naming inconsistency in messages, not as a change in algorithmic behavior.

## Scientific Objective

The stage takes a training HDF5 file, typically `TRAIN.h5`, and produces a reduced training HDF5, typically `TRAIN_FILTERED.h5`.

Its scientific intent is:

1. Reduce the number of training patches.
2. Keep patch diversity within each patient.
3. Avoid cross-patient mixing during selection.
4. Preserve training-file compatibility for downstream stages.

This is an unsupervised, patient-wise redundancy reduction algorithm. It does not optimize directly for classification accuracy, segmentation quality, or label preservation. It optimizes for embedding-space coverage and patient-wise diversity.

## High-Level Logic

For each patient independently, the algorithm:

1. Builds image embeddings for some or all of that patient's patches.
2. Uses a stability heuristic to decide how many patches need embeddings.
3. Clusters the chosen embeddings with MiniBatch K-Means.
4. Builds a cluster-balanced ordering of patches.
5. Optionally finds a smaller keep count by detecting when coverage improvement plateaus.
6. Keeps only the selected global patch rows.
7. Writes a new HDF5 containing only those kept rows.

At no point does it use labels or masks to decide what to keep.

## Exact Pipeline Behavior

### 1. Entrypoint behavior

`7_smart_sampler.py` does only orchestration:

1. Load `.env` with override enabled.
2. Parse environment variables into `SmartSamplerConfig`.
3. Configure logging.
4. Seed randomness through `seed_everything(config.seed)`.
5. Clear CUDA cache if available.
6. Run `run_smart_sampling_pipeline(config)`.
7. Print and log summary statistics.
8. Exit with code `0` on success or `2` on failure.

The actual selection algorithm lives in the helper modules.

### 2. Configuration parameters

The key algorithmic parameters are:

- `source_h5_path`: input training HDF5.
- `output_dir`, `output_filename`: where the filtered HDF5 is written.
- `encoder_name`, `encoder_weights`, `input_size`, `batch_size`, `device`, `num_workers`: embedding extractor settings.
- `n_start`: initial number of patches to embed per patient.
- `n_max`: maximum number of patches to embed per patient.
- `growth_factor`: multiplier used when stability is too low.
- `stability_threshold`: ARI threshold used to accept a sampled embedding subset.
- `max_steps`: maximum number of growth steps.
- `intersection_ratio_threshold`: decides whether stability is computed on the intersection of two subsets or on one subset only.
- `k_min`, `k_max`: lower and upper bounds for clustering count.
- `adaptive_keep_enabled`: whether to use plateau-based keep-count selection.
- `keep_min`, `keep_step`, `keep_improvement_threshold`, `keep_patience`: adaptive keep controls.
- `m_max`: hard upper limit on the number of patches kept per patient.
- `seed`: master deterministic seed.

Two notable transparency issues:

1. `stability_repeats` exists in config but is not used by the selection code.
2. `selection_strategy` exists in config but is not used to switch behavior.

### 3. HDF5 indexing

The pipeline first reads the source HDF5 and verifies that these datasets exist:

- `images`
- `masks`
- `patient_ids`
- `labels`
- one of `filenames` or `filename`

It then creates a mapping:

- `patient_id -> array of global row indices`

All downstream sampling is performed separately for each patient using these global row indices.

This is important scientifically because it preserves patient-wise isolation during sampling.

### 4. Embedding extraction

The algorithm embeds image patches using the encoder part of a `segmentation_models_pytorch.Unet` model.

Embedding extractor behavior:

1. Construct a U-Net with the configured encoder, usually something like `resnet50` with ImageNet weights.
2. Throw away the decoder and keep `model.encoder` only.
3. For each requested HDF5 row index, load the `images` dataset row.
4. Convert channel-first images to channel-last if needed.
5. Apply preprocessing:
   - `ToPILImage()`
   - resize to `(input_size, input_size)`
   - `ToTensor()`
   - ImageNet normalization with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
6. Run the batch through the encoder.
7. Take the last encoder feature map.
8. Apply global average pooling over spatial dimensions.
9. Use the pooled vector as the patch embedding.

So each patch is represented by one fixed-length float vector derived from the final encoder feature map.

### 5. Per-patient subset-size search

For each patient, let `patient_indices` be all global patch rows for that patient.

The algorithm decides how many of those rows need to be embedded before final selection.

#### 5.1 Small-patient shortcut

If:

`patch_count <= 1.5 * n_start`

then the algorithm skips the stability search and embeds all patches for that patient.

#### 5.2 Stability-driven subset search

If the patient has more patches than that threshold:

1. Start with `n_curr = min(n_start, patch_count)`.
2. Create a random generator seeded with `config.seed + patient_id`.
3. Repeat up to `max_steps` times:
   - Sample two different subsets of size `n_curr` without replacement from the patient's full patch set. Call them `S1` and `S2`.
   - Compute embeddings for both subsets, using a cache so repeated rows are embedded only once.
   - Compute a clustering stability score between the two subsets.
   - If the score is at least `stability_threshold`, accept `S1` as the candidate pool and stop.
   - Otherwise grow `n_curr = int(n_curr * growth_factor)`.
   - If growth would exceed `n_max` or the patient size, stop and use one final random subset of size `min(patch_count, n_max)`.

If no branch has set embeddings by the end, it falls back to a random subset of size `min(patch_count, n_start)`.

#### 5.3 How stability is computed

Given embeddings for `S1` and `S2`:

1. Compute `k = compute_k(n_samples)`.
2. Fit one MiniBatch K-Means on `S1` with `random_state = seed`.
3. Fit a second MiniBatch K-Means on `S2` with `random_state = seed + 1`.
4. If the overlap size between `S1` and `S2` is greater than `intersection_ratio_threshold * n_samples`:
   - Restrict to the common rows.
   - Predict cluster labels for the common embeddings under both clusterers.
   - Return the Adjusted Rand Index between the two labelings.
5. Otherwise:
   - Use cluster labels from the first clusterer on `S1`.
   - Predict labels for `S1` using the second clusterer.
   - Return the Adjusted Rand Index between those two labelings.

Interpretation:

If two random subsets of this patient produce similar cluster structure, the algorithm assumes the patient manifold is already well represented and there is no need to embed more patches.

### 6. Choosing the number of clusters

The helper `compute_k(n_samples, config)` uses:

1. `raw_k = floor(sqrt(n_samples))`
2. `k = clamp(raw_k, k_min, k_max)`

Special handling for very small `n_samples`:

- If `n_samples < k_min`, set `k = max(2, floor(n_samples / 10))`.
- If that drops below `2`, use `k = n_samples`.

Implications:

- Medium and large pools get about `sqrt(n)` clusters, bounded by `k_min` and `k_max`.
- Very small pools can end up with tiny cluster counts.
- `k` is data-size driven, not performance-driven.

### 7. Final diverse-sample selection inside the candidate pool

After the candidate pool is chosen for a patient, the algorithm keeps at most:

`m_target = min(m_max, patch_count)`

It never keeps more than `m_max` rows per patient.

Then it performs clustering and selection.

#### 7.1 Cluster the candidate pool

1. Compute `k = compute_k(len(candidate_pool), config)`.
2. Fit MiniBatch K-Means with that `k` on the candidate embeddings.
3. Obtain one cluster label per candidate patch.

#### 7.2 If adaptive keep is disabled

The algorithm performs quota-based uniform cluster sampling:

1. Set `quota = ceil(m_target / k)`.
2. For each cluster:
   - If cluster size is at most `quota`, keep all rows in that cluster.
   - Otherwise randomly sample exactly `quota` rows from that cluster with seed `config.seed`.
3. If the total kept is still below `m_target`, take extra rows from an overflow pool in sorted-index order.
4. Sort the final selected indices.

This mode tries to spread selection across clusters.

#### 7.3 If adaptive keep is enabled

The algorithm first builds a cluster-balanced ordering, then truncates it.

##### 7.3.1 Build the cluster-balanced ordering

1. Group candidate rows by cluster label.
2. Shuffle rows inside each cluster using a deterministic RNG with `seed`.
3. Perform round-robin extraction across clusters:
   - take one row from cluster 0
   - then one from cluster 1
   - and so on
   - continue until all clusters are exhausted

This yields `selection_order`, a deterministic interleaving of clusters.

##### 7.3.2 Score partial prefixes by coverage

For candidate keep counts:

- start at `min_keep = min(max_keep, keep_min)`
- increase by `keep_step`
- stop at `max_keep = min(m_target, candidate_pool_size)`

For each prefix length `s` of `selection_order`, compute the coverage score:

1. Take the first `s` selected indices.
2. Gather their embeddings.
3. Compute Euclidean distance from every candidate embedding to every selected embedding.
4. For each candidate embedding, find the nearest selected embedding.
5. Average those nearest-neighbor distances.

Lower score means the selected subset covers the full candidate embedding cloud better.

##### 7.3.3 Plateau stopping rule

The algorithm tracks relative improvement in coverage score:

`relative_improvement = (previous_score - current_score) / max(previous_score, 1e-12)`

If improvement falls below `keep_improvement_threshold`, increment a plateau counter.

If plateau counter reaches `keep_patience`, stop and choose the previous keep count.

So adaptive keep finds the smallest prefix after which additional selected patches no longer improve embedding-space coverage enough.

### 8. Global output creation

After all patients are processed:

1. Merge all selected global row indices across patients.
2. Deduplicate them.
3. Sort them globally.
4. Compute summary statistics:
   - total input rows
   - kept rows
   - rejected rows
   - kept fraction
   - number of patients
   - number of patients that were reduced

### 9. Writing the filtered HDF5

The output writer creates a new HDF5 with these datasets:

- `images`
- `masks`
- `patient_ids`
- `labels`
- `filenames`

If the input uses `filename`, the output still standardizes to `filenames`.

Rows are copied in sorted selected-index order.

The writer also stores provenance attributes such as:

- `selection_signature`
- source HDF5 SHA-256
- some upstream cleaning/signature attributes if present

This is good for reproducibility and artifact tracking.

### 10. Sidecar artifacts

If enabled, the stage writes:

- `train_filtered_selection.csv`: one row per kept patch with patient id, global index, embedding subset size, cluster count, and selection method.
- `patient_filter_stats.csv`: one row per patient with total patches, selected count, runtime, stability trace, and retention trace.
- `filter_run_config.json`: serialized config for the run.
- `filter_summary.json`: top-level summary.

These files matter because the HDF5 alone is not enough to understand why rows were kept.

## Pseudocode for Full Replication

```text
load environment
parse config
seed all RNGs

index input HDF5 by patient_id -> global row indices
initialize embedding extractor

selected_global_indices = []
selection_manifest = []
stats_log = []

for each patient_id in sorted patient ids:
    patient_indices = indices for this patient
    patch_count = len(patient_indices)
    rng = default_rng(seed + patient_id)
    embedding_cache = {}
    stability_history = []

    define fetch_embeddings(indices):
        embed only missing rows
        cache embeddings by global row index
        return embeddings in requested order

    if patch_count <= 1.5 * n_start:
        candidate_pool = patient_indices
        final_embeddings = fetch_embeddings(candidate_pool)
        stability_history.append((patch_count, 1.0))
    else:
        n_curr = min(n_start, patch_count)
        for step in range(max_steps):
            if n_curr >= patch_count:
                candidate_pool = patient_indices
                final_embeddings = fetch_embeddings(candidate_pool)
                break

            S1 = sorted random subset of patient_indices of size n_curr
            S2 = sorted random subset of patient_indices of size n_curr
            E1 = fetch_embeddings(S1)
            E2 = fetch_embeddings(S2)
            score = clustering_stability(E1, E2, S1, S2)
            stability_history.append((n_curr, score))

            if score >= stability_threshold:
                candidate_pool = S1
                final_embeddings = E1
                break

            new_n = int(n_curr * growth_factor)
            if new_n >= n_max or new_n >= patch_count:
                n_curr = min(patch_count, n_max)
                candidate_pool = sorted random subset of patient_indices of size n_curr
                final_embeddings = fetch_embeddings(candidate_pool)
                break
            n_curr = new_n

    if final_embeddings still not set:
        candidate_pool = sorted random subset of size min(patch_count, n_start)
        final_embeddings = fetch_embeddings(candidate_pool)

    m_target = min(m_max, patch_count)
    cluster final_embeddings using MiniBatchKMeans with k = bounded sqrt(n)

    if adaptive_keep_enabled:
        build cluster-balanced round-robin ordering
        for keep_count from keep_min to max_keep step keep_step:
            compute mean nearest-selected distance over all candidate embeddings
            stop when relative improvement plateaus for keep_patience steps
        selected_patient_indices = prefix of ordering at chosen keep count
    else:
        assign per-cluster quota = ceil(m_target / k)
        sample quota rows per cluster
        fill any deficit from overflow rows
        selected_patient_indices = sorted chosen rows

    append selected_patient_indices to selected_global_indices
    log per-patient stats and per-kept-row manifest

deduplicate and sort all selected_global_indices
copy only those rows into output HDF5
write sidecar CSV/JSON artifacts
```

## What This Algorithm Is Good At

### 1. Removing within-patient redundancy

If a patient contributes many visually similar patches, the algorithm can reduce that redundancy instead of training repeatedly on near-duplicates.

### 2. Preserving patient-wise boundaries

Selection is done patient by patient. This is scientifically safer than selecting globally across the entire training set, because one patient cannot dominate the representation budget of another patient.

### 3. Keeping morphological diversity without labels

The clustering and coverage heuristics aim to preserve a spread of embedding-space phenotypes even when no supervision is used.

### 4. Saving compute and storage

If training is expensive, this stage can materially reduce downstream time and disk usage.

### 5. Adding provenance

The pipeline records run config, patient stats, selection manifest, and output signature, which helps reproducibility and auditability.

## What This Algorithm Is Not Good At

### 1. Preserving rare but important pathology by design

Because selection is unsupervised and label-agnostic, the algorithm can discard rare but clinically critical patches if they are not well protected by the embedding-space diversity heuristic.

### 2. Optimizing for downstream task performance

The objective is coverage in encoder embedding space, not classification AUC, segmentation Dice, calibration, or patient-level outcome quality.

### 3. Respecting class balance automatically

It does not explicitly preserve:

- cancer vs not-cancer prevalence
- lesion-size distribution
- hard-negative prevalence
- mask-positive pixel prevalence
- site or scanner distribution

### 4. Explaining biological meaning

Clusters are formed in encoder feature space. Those clusters are computational groupings, not validated biological or pathological phenotypes.

### 5. Handling domain shift robustly

The embedding model uses a generic encoder with ImageNet-style preprocessing. If the pathology domain differs strongly from the pretraining domain, similarity judgments may be misaligned with medically relevant structure.

## Scientific Review Findings

Findings are ordered by severity.

### Critical

No critical flaws were found that automatically imply train/validation/test leakage inside this stage, assuming the input is truly a training-only HDF5.

### Major

#### 1. Label-agnostic filtering can remove rare positive evidence

Severity: Major

The sampler never uses `labels` or `masks` when deciding what to keep. It only uses image embeddings and cluster coverage. In medical datasets, rare positives or subtle lesion morphologies are often exactly the patches least protected by frequency-based or coverage-based reduction.

Scientific risk:

- positive-class prevalence can be reduced unpredictably
- rare disease patterns may be removed
- reported performance can become biased toward common morphologies
- external validity can degrade if uncommon but important patterns disappear

Correction:

- audit class counts before and after filtering
- audit mask-positive area before and after filtering
- optionally enforce stratified keep rules by label or lesion burden
- compare downstream results against the unfiltered training set

#### 2. The method optimizes embedding coverage, not task utility

Severity: Major

The plateau rule uses mean nearest-selected distance in embedding space. That is a representation-compression objective, not a clinical or predictive objective.

Scientific risk:

- a patch can be redundant in encoder space but still highly useful for model calibration or decision boundaries
- the reduced set may look diverse while harming task-relevant supervision density
- conclusions about "better data quality" would be scientifically unsupported unless validated downstream

Correction:

- treat this as a data-reduction heuristic, not a validated improvement method
- report downstream performance deltas with confidence intervals
- evaluate whether class-specific sensitivity changes after filtering

#### 3. Stability search is based on a single subset-pair per step

Severity: Major

The config exposes `stability_repeats`, but the code computes only one stability score per subset size. That makes the selected embedding-pool size sensitive to one random draw of `S1` and `S2`.

Scientific risk:

- unstable keep decisions across runs or seeds
- overconfidence if users believe multiple repeats were averaged
- reduced reproducibility of reported reduction rates

Correction:

- either implement repeated subset-pair sampling and average the ARI
- or remove `stability_repeats` from config and clearly document that only one comparison is used

### Moderate

#### 4. Domain mismatch risk from generic pretrained encoder features

Severity: Moderate

Embeddings come from the encoder of a generic U-Net with ImageNet-style normalization. Those embeddings may not reflect pathology-relevant similarity well.

Scientific risk:

- visually similar embeddings may not correspond to clinically similar tissue states
- cluster structure may emphasize texture or color artifacts over pathology

Correction:

- validate the embedding space qualitatively and quantitatively
- compare with a pathology-domain encoder or task-trained encoder
- inspect class retention and lesion retention by cluster

#### 5. `m_max` enforces a hard per-patient cap regardless of patient complexity

Severity: Moderate

Every patient can contribute at most `m_max` selected patches even if one patient is much more heterogeneous than another.

Scientific risk:

- complex or lesion-rich patients may be underrepresented
- simple patients and complex patients can be forced toward similar retained counts

Correction:

- benchmark patient-adaptive caps
- report retained fraction as a function of original patient patch count and lesion burden

#### 6. Coverage scoring is computed on the same candidate pool used to construct the ordering

Severity: Moderate

The algorithm evaluates prefix coverage on the same embeddings from which clusters and ordering were derived.

Scientific risk:

- optimistic estimate of how well the subset represents that candidate pool
- no external check that retained samples preserve clinically relevant variability

Correction:

- validate with downstream performance
- or evaluate retained subsets against held-out patient-internal patches when feasible

### Minor

#### 7. `selection_strategy` is unused

Severity: Minor

The config suggests selectable strategies, but the code path does not branch on it.

Scientific risk:

- method reporting can become inaccurate if users think a different strategy was active

Correction:

- either implement strategy switching or remove the unused parameter

#### 8. Message naming is inconsistent about stage number

Severity: Minor

Logs mention both Stage 7 and Stage 8.

Scientific risk:

- audit trails become harder to interpret

Correction:

- standardize naming in logs and filenames

## Pros and Cons

### Pros

- Patient-wise selection reduces the chance that one patient dominates globally.
- Embedding caching avoids redundant feature extraction within a patient.
- The stability heuristic can avoid embedding every patch for very large patients.
- Cluster-balanced ordering encourages coverage across multiple embedding modes.
- Adaptive plateau stopping can select fewer than `m_max` patches when additional patches add little coverage.
- Provenance sidecars and signatures make the output more auditable than an opaque filtered file.

### Cons

- No explicit guarantee of preserving rare positives, lesion borders, or hard negatives.
- Embedding quality depends heavily on the pretrained encoder choice.
- Stability is estimated from one random pair of subsets, not repeated measurements.
- The adaptive coverage metric is purely geometric and may not align with task value.
- Hard patient caps can underrepresent highly heterogeneous patients.
- The adaptive coverage calculation is computationally expensive because it uses dense pairwise distances.
- Several config fields imply capabilities that the current code does not actually use.

## Reproducibility Assessment

### What already helps reproducibility

- Global seeding is performed.
- Per-patient RNG is deterministic through `seed + patient_id`.
- Output selection is recorded in CSV.
- Full run config is written to JSON.
- Output HDF5 stores a selection signature and source provenance hash.

### Remaining reproducibility risks

- GPU inference can still have backend nondeterminism depending on the wider training runtime settings.
- The undocumented non-use of `stability_repeats` can mislead experiment tracking.
- If encoder weights change remotely or by package version, the embedding space changes and so does the selected subset.

## Recommended Reporting If This Is Used In Research

Any paper, report, or benchmark using this stage should state clearly that:

1. Filtering is unsupervised and patient-wise.
2. Selection is based on encoder embedding diversity, not labels.
3. A hard per-patient cap is applied.
4. The keep count may be further reduced by a coverage plateau heuristic.
5. The resulting subset must be validated by downstream task metrics, not assumed superior.

At minimum, report before/after:

- total patch count
- per-patient retained fraction
- class counts
- mask-positive area distribution
- downstream performance with uncertainty estimates

## Bottom Line

This is a reasonable engineering approach for patient-wise redundancy reduction in a large patch dataset. It is strongest when the main goal is to compress training data while retaining broad morphological diversity.

It is not, by itself, evidence that the resulting training set is scientifically better. The method is unsupervised, label-agnostic, and driven by generic embedding geometry. For scientific claims, it must be treated as a heuristic preprocessing step whose effects on class retention, lesion retention, and downstream performance are explicitly validated.
