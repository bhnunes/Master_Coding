2. Algorithmic & Statistical Fixes (Addressing Moderate Risks)

The internal logic has a few statistical brittle points that undermine the "smart" aspect of the sampler.

a) Enforce Stability Repeats: The review noted that stability_repeats is in the config but ignored in the code. Fix this. A single subset pair ($S1$ and $S2$) is statistically unsafe. Update the code to loop over the subset pairing $N$ times, compute the Adjusted Rand Index ($\text{ARI}$) for each, and require the mean $\text{ARI}$ to pass the stability_threshold.

b) Implement a Dynamic, Patient-Adaptive Cap: Replace the hard $m_{max}$ limit. A patient with a highly heterogeneous tumor should have a larger representation budget than a patient with entirely homogeneous healthy tissue. Make $m_{max}$ scale dynamically based on the optimal $k$ found during the stability check, or based on the variance of the patient's embeddings.

c) Fix Circular Coverage Scoring: Currently, the plateau stopping rule evaluates coverage on the exact same pool used to define the clusters. To prevent optimistic bias, implement a cross-validation approach: calculate the coverage score against a held-out fraction of the patient's patches that were not in the candidate pool.

3. Codebase Health & Transparency (Addressing Minor Risks)

Technical debt and misleading logs will erode trust in the pipeline's outputs.

d) Remove or Implement Dead Code: The selection_strategy parameter is a broken promise in the configuration. Either build out the alternative strategies (e.g., replacing uniform sampling with density-based sampling) or delete the variable from helpers/smart_sampling/config.py to ensure the codebase matches the actual math.

e) Unify the Naming Convention: The "Stage 7 vs. Stage 8" inconsistency is a minor but irritating bug that ruins audit trails. Standardize the logger.info statements across the orchestrator and all helper modules to read "Stage 7" uniformly.

f) Enhance the Provenance Sidecar: In train_filtered_selection.csv, explicitly log the exact threshold at which the plateau rule triggered for each patient. This allows researchers to audit why the algorithm stopped sampling, rather than just seeing the final count.