Systematic Review Protocol: Data Governance, Reproducibility, and End-to-End Methodologies in Cancer Digital Pathology WSI Pipelines

1. Primary Research Questions (RQs)
RQ1 (Artifacts & QC): How do recent end-to-end WSI pipelines handle slide-level artifacts, background contamination, and necrosis prior to deep learning patch extraction?
RQ2 (Data Leakage & Preprocessing): How does the current literature address patient-level dataset partitioning and the risk of train-test leakage during dataset-wide preprocessing (e.g., stain normalization)?
RQ3 (Data Governance & Provenance): To what extent do contemporary pipelines implement explicit data contracts (e.g., HDF5-native workflows, Parquet sidecars), database-backed orchestration, and automated provenance tracking?
RQ4 (Ensembling): What strategies are utilized for multi-model aggregation in histopathology segmentation, and how is validation-only recipe optimization managed compared to naive pixel-averaging?

2. Eligibility Criteria (PICOS Adapted)
Inclusion Criteria:
Domain: Application of Deep Learning to Whole-Slide Histopathology.
Biological Target: Must explicitly focus on the detection, segmentation, or classification of cancer/neoplasms/tumors.
Methodological Focus: Must explicitly detail the pipeline infrastructure for at least one of the following: slide-level quality control, stain normalization, patient-level data partitioning, data serialization (e.g., HDF5, Zarr), or multi-model ensembling.
Reproducibility: Must provide a link to a public repository (GitHub, GitLab, etc.) containing the pipeline implementation.
Publication Parameters: Published between January 1, 2020, and December 31, 2025; English language; Open Access.
Exclusion Criteria:
Target Flaws: Purely clinical correlation studies or focus on non-neoplastic conditions.
"Pre-Extracted Patch" Scam: Studies that bypass WSI processing completely by utilizing perfectly curated, pre-extracted patch datasets (e.g., standard PatchCamelyon, CRC-100K).
"Obvious Data Leakage" Flaw: Studies that split data at the patch or slide level (ignoring patient ID) or apply global dataset stain normalization before generating train/validation/test splits.
"Architecture-Only" Trap: Papers whose primary novelty is tweaking a neural network architecture while treating data extraction, provenance, and pipeline orchestration as an undocumented black box.
"Trust-Me Science": Closed-source papers lacking a publicly verifiable codebase.

3. Databases and Exact Search Queries
A. PubMed / MEDLINE
("Whole Slide Imaging"[MeSH] OR "histopathology"[tiab] OR "whole-slide"[tiab] OR "WSI"[tiab]) AND ("Neoplasms"[MeSH] OR "cancer"[tiab] OR "tumor"[tiab] OR "tumour"[tiab] OR "carcinoma"[tiab] OR "malignancy"[tiab]) AND ("Deep Learning"[MeSH] OR "deep learning"[tiab] OR "ensemble learning"[tiab] OR "neural networks"[tiab] OR "segmentation"[tiab]) AND ("reproducibility"[tiab] OR "artifact"[tiab] OR "stain normalization"[tiab] OR "quality control"[tiab] OR "patient-level"[tiab] OR "data leakage"[tiab] OR "pipeline"[tiab]) AND ("2020/01/01"[Date - Publication] : "2025/12/31"[Date - Publication])
B. IEEE Xplore
("Document Title":histopathology OR "Document Title":"whole slide" OR "Document Title":WSI OR "Abstract":histopathology) AND ("Abstract":cancer OR "Abstract":tumor OR "Abstract":tumour OR "Abstract":carcinoma OR "Abstract":malignancy) AND ("Abstract":"deep learning" OR "Abstract":"ensemble" OR "Abstract":"segmentation") AND ("Abstract":reproducibility OR "Abstract":artifact OR "Abstract":normalization OR "Abstract":"data leakage" OR "Abstract":"patient-level" OR "Abstract":"quality control" OR "Abstract":"pipeline")
C. SpringerLink
("whole slide" OR "WSI" OR "computational pathology") AND ("cancer" OR "tumor" OR "carcinoma") AND ("deep learning" OR "convolutional neural network") AND ("stain normalization" OR "reproducibility" OR "artifact" OR "data leakage" OR "quality control")

4. Two-Phase Screening Workflow & LLM Prompts
All records will be deduplicated using a reference manager. Screening will occur in two strict phases using the following LLM prompts to assist reviewers.

Phase 1: Basic Relevance & Leakage Check (Abstract/Methodology)

```
You are an expert, highly critical academic reviewer specializing in Deep Learning applied to Digital Pathology and Whole-Slide Imaging (WSI). Your task is to review scientific papers (abstracts and methodology sections) and decide if they should be included in a rigorous systematic literature review. 

The focus of this review is on end-to-end computational pipelines for cancer detection, with a strict emphasis on methodological rigor, data leakage prevention, artifact handling, and reproducibility.

EVALUATION CRITERIA:
1. ACCEPTANCE (Must meet ALL of these):
- CANCER FOCUS: The paper explicitly applies deep learning to the detection, segmentation, or classification of cancer/neoplasms/tumors.
- WSI PIPELINE: The paper processes Whole-Slide Images (WSI) and explicitly details the infrastructure/methodology for at least one of the following: slide-level quality control/artifact rejection, stain normalization algorithms, data partitioning strategies, or multi-model ensembling.

2. REJECTION (Reject if ANY of these apply):
- NON-ONCOLOGICAL / PURE CLINICAL: The paper focuses on non-cancerous diseases or is purely a biomarker study lacking DL pipeline methodology.
- PRE-EXTRACTED PATCH SCAM: The paper completely bypasses WSI challenges by utilizing pre-extracted patch datasets (e.g., CRC-100K). It must detail slide-to-patch extraction.
- OBVIOUS DATA LEAKAGE: The paper splits data at the patch/slide level instead of the patient level, or applies dataset-wide stain normalization before splitting.
- METHODOLOGICAL BLACK BOX: Fails to explain how they handle WSI splitting, background artifacts, or reproducibility.

INSTRUCTIONS:
Internally analyze the paper's target disease, WSI processing, artifact handling, dataset partitioning, and stain normalization. Identify if it falls into any Rejection trap. 
Output your reasoning in a short paragraph.
Your output MUST end with exactly one of the following lines:
FINAL VERDICT: ACCEPTED PHASE 1
FINAL VERDICT: REJECTED PHASE 1
```

Phase 2: Strict Engineering & Reproducibility Filter (Full-Text)
```
You are an expert, highly critical academic reviewer specializing in Deep Learning applied to Digital Pathology. You are evaluating a shortlist of papers that have passed a basic relevance check. Your goal is to ruthlessly filter this list down to only the highest-quality papers focusing on computational reproducibility and rigorous data pipeline engineering.

EVALUATION CRITERIA:
1. ACCEPTANCE (Must meet ALL of these):
- TRANSPARENT REPRODUCIBILITY: Provides a link to a public repository (GitHub, GitLab, etc.) for their pipeline. 
- EXPLICIT DATA GOVERNANCE: Explicitly details its data-engineering layer (e.g., mentions HDF5, Zarr, SQLite, Parquet, spatial databases, or strict dataset manifests). 

2. REJECTION (Reject if ANY of these apply):
- CLOSED SOURCE / TRUST-ME SCIENCE: Claims an end-to-end pipeline but does not publish the code. 
- ARCHITECTURE OVER-INDEXING: The novelty is purely tweaking a neural network architecture while treating data extraction/provenance tracking as an afterthought.
- NAIVE PREPROCESSING/QC: Handles artifacts using only basic hardcoded thresholds (e.g., standard Otsu) without advanced, human-in-the-loop, or dynamic scoring.
- TRIVIAL ENSEMBLING: If ensembling is used, it relies on naive majority-voting/averaging without a dedicated spatial WSI optimization/calibration phase.

INSTRUCTIONS:
Internally scan the paper for mentions of public repositories, HDF5/Database usage, and advanced pipeline logistics.
Output your reasoning in a harsh, critical, short paragraph, pointing out exactly which criteria it failed.
Your output MUST end with exactly one of the following lines:
FINAL VERDICT: ACCEPTED PHASE 2
FINAL VERDICT: REJECTED PHASE 2

```

5. Data Extraction & Synthesis
For all papers surviving Phase 2, a data extraction matrix will be populated to map directly to the Related Work sections of the manuscript:
Reference / Year.
Artifact/QC Strategy: (e.g., Fixed Thresholds, External CNN, Graph-based, Human-in-the-loop).
Normalization Policy: (e.g., Global Macenko[Leakage], Train-only template, None).
Data Partitioning: (e.g., Patch-level [Leakage], Slide-level [Leakage], Verified Patient-level).
Storage / Governance: (e.g., PNG folders, LMDB, HDF5, Zarr, SQLite-backed).
Ensembling / Post-processing: (e.g., None, Pixel-averaging, Multi-stream gating).
Code Availability: (Link to repository).
