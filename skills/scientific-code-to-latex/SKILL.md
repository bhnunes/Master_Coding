---
name: scientific-code-to-latex
description: >
  Use when converting research code or repositories into publication-ready
  scientific prose in LaTeX, including Methods sections, experimental setup,
  architecture descriptions, or reproducibility documentation grounded in the
  actual implementation.
compatibility: opencode
---

# Scientific Code-to-LaTeX Writer

This skill enables the agent to transform a software or research
codebase into publication-ready scientific prose in LaTeX.

The generated text must be grounded in the **actual implementation,
experimental design, and reproducibility constraints** of the project.

The goal is to translate code artifacts into **manuscript-quality
scientific text** without inventing methodology or unsupported claims.

---

## When to Use
Use this skill when the user asks for any of the following:
- convert code into scientific writing,
- draft Methods / Materials and Methods / Experimental Setup sections,
- explain a pipeline, architecture, or algorithm from source code,
- generate LaTeX text for a paper, dissertation, report, or appendix,
- document reproducibility, training, inference, evaluation, or ablation procedures from code.

Do **not** use this skill when:
- the repository has not been inspected,
- the requested claims are not supported by the code or accompanying documentation,
- the user asks for speculative scientific conclusions that cannot be justified from the implementation or evidence.

---

## Core Principles
1. **Code-grounded only**  
   Every scientific statement must be traceable to the code, configuration, logs, or user-provided context.

2. **No invention**  
   Never fabricate datasets, hyperparameters, equations, baselines, metrics, experimental procedures, or results.

3. **Paper style, not code style**  
   Convert implementation details into concise scientific prose, emphasizing rationale, methodology, reproducibility, and limitations.

4. **LaTeX-ready output**  
   Output must be directly usable in a LaTeX manuscript, with proper escaping of reserved characters when needed.

5. **Explicit uncertainty**  
   If some scientific detail cannot be verified from the codebase, state the gap clearly and mark it as a placeholder or unresolved item.

6. **Reproducibility first**  
   Prefer describing data flow, preprocessing, model definition, training procedure, inference procedure, evaluation metrics, seeds, hardware assumptions, and dependency constraints.

---

## Required Inputs
The agent should gather as much of the following as possible before writing:
- repository structure,
- main entrypoints,
- configuration files,
- training scripts,
- inference scripts,
- evaluation scripts,
- utility modules,
- experiment logs or run metadata,
- README / docs / comments,
- user-provided scientific context.

If available, also inspect:
- `requirements.txt`, `pyproject.toml`, `environment.yml`, `Dockerfile`,
- YAML / JSON config files,
- notebooks used for experiments,
- LaTeX manuscript structure,
- result tables or figures,
- dataset manifest files.

---

## Recommended Workflow

### Step 1 — Map the Repository
Identify:
- the scientific objective of the codebase,
- the main computational pipeline,
- inputs and outputs of each stage,
- which scripts are authoritative,
- where hyperparameters and defaults are defined.

Produce an internal map such as:
- data acquisition / loading,
- preprocessing,
- feature extraction,
- model architecture,
- optimization / training,
- validation / selection,
- inference,
- post-processing,
- evaluation,
- logging / reproducibility.

### Step 2 — Extract Verifiable Facts
For each relevant module, extract only facts supported by the implementation, such as:
- algorithm names,
- architecture blocks,
- tensor/data flow,
- loss functions,
- metrics,
- sampling strategy,
- augmentation policy,
- optimizer and scheduler,
- thresholding or calibration,
- ensembling logic,
- file formats,
- deterministic settings,
- hardware/distributed assumptions.

### Step 3 — Infer the Scientific Role
Translate implementation into its methodological role. For example:
- `Dataset` classes → data representation and sampling strategy,
- transforms/augmentations → preprocessing and robustness strategy,
- model modules → architecture and information flow,
- training loop → optimization protocol,
- evaluation script → assessment methodology,
- post-processing utilities → decision refinement / filtering.

Only make inferences that are strongly supported by the code.

### Step 4 — Write in Scientific Form
Convert code-level detail into manuscript-level prose.

Examples:
- From: `AdamW(lr=1e-4, weight_decay=1e-2)`  
  To: `Model parameters were optimized using AdamW with an initial learning rate of $10^{-4}$ and weight decay of $10^{-2}$.`

- From: `Dice + BCE hybrid loss`  
  To: `Training employed a hybrid objective combining Dice loss and binary cross-entropy to balance region overlap optimization with pixel-wise supervision.`

- From: `patient_id split constraint`  
  To: `Data partitioning was performed at the patient level to prevent information leakage across training, validation, and test sets.`

### Step 5 — Produce LaTeX-Safe Text
Ensure the output:
- uses proper LaTeX sectioning when requested,
- escapes reserved characters such as `_`, `%`, `&`, `#` when outside math/code contexts,
- preserves math notation in proper LaTeX form,
- uses manuscript tone rather than bullet-point engineering notes unless explicitly requested.

### Step 6 — Add Scientific Safeguards
Before finalizing, verify:
- no unsupported causal claims were introduced,
- no results are described unless explicitly available,
- no novelty claims are made without user confirmation,
- no architecture names or citations are guessed,
- no reproducibility-critical detail was omitted when available in code.

---

## Output Modes
Depending on the request, the agent may generate one or more of the following:

### 1. Methods Section
A polished LaTeX subsection or section describing methodology.

### 2. Architecture Description
Paper-style explanation of the model or system design.

### 3. Experimental Setup
LaTeX text covering data splits, hardware, software, hyperparameters, losses, metrics, and training protocol.

### 4. Reproducibility Appendix
Structured appendix text describing seeds, packages, configs, checkpoints, and execution flow.

### 5. Figure/Table Captions
Short scientific captions grounded in actual outputs.

### 6. Traceability Notes
A side artifact mapping each paragraph to the source files that support it.

---

## Preferred Section Template
When writing a scientific section from code, use this structure when applicable:

1. **Objective** — what the module/pipeline is intended to do.
2. **Inputs and outputs** — what enters and what is produced.
3. **Method** — the algorithmic or architectural procedure.
4. **Optimization / execution** — how it is trained or run.
5. **Evaluation / decision logic** — how outputs are assessed or selected.
6. **Reproducibility details** — seeds, configs, dependencies, hardware, file formats.
7. **Limitations / unresolved gaps** — anything not verifiable from code.

---

## Writing Style Requirements
- Use formal scientific English unless the user requests another language.
- Prefer precise, compact sentences.
- Define technical terms when first introduced if needed.
- Avoid marketing language.
- Avoid exaggerated novelty claims.
- Avoid saying the code is “robust”, “state-of-the-art”, or “highly accurate” unless evidence is provided.
- Prefer passive voice or neutral scientific voice when suitable.
- Keep terminology consistent with the implementation.

---

## LaTeX Conventions
When generating LaTeX:
- Use standard section commands such as `\section{}`, `\subsection{}`, `\paragraph{}` only if the user requests structural markup.
- Escape special characters in normal text.
- Put mathematical symbols in math mode.
- Do not generate BibTeX citations unless the user asks for them.
- Do not invent `\cite{}` keys.
- Keep code identifiers in `\texttt{}` when useful.

Example:
```latex
\subsection{Training Procedure}
Model parameters were optimized using \texttt{AdamW} with an initial learning rate of $10^{-4}$ and a weight decay of $10^{-2}$. To mitigate information leakage, all dataset partitions were generated at the patient level.
```

---

## Scientific Validation Checklist
Before returning text, the agent must check:
- Is every methodological statement supported by the repository or user context?
- Are data splits described correctly?
- Are hyperparameters copied faithfully?
- Are losses, metrics, and thresholds named correctly?
- Are model components described in the right order?
- Is the difference between training, validation, and test procedures preserved?
- Are post-processing steps distinguished from core model behavior?
- Are assumptions or unknowns explicitly flagged?
- Is the text suitable for a paper rather than a README?
- Is the output LaTeX-safe?

---

## Failure Conditions
The skill must stop and warn the user when:
- the codebase is too incomplete to support the requested scientific text,
- the repository contradicts the claimed methodology,
- crucial experimental details are missing,
- the user asks for fabricated results or unsupported claims,
- citation placeholders would require guessing references.

In such cases, return:
1. what could be verified,
2. what could not be verified,
3. a minimal safe draft with clearly marked placeholders.

---

## Example Requests This Skill Should Handle
- `Convert my training pipeline into a Methods section in LaTeX.`
- `Read this segmentation codebase and draft the experimental setup for my paper.`
- `Explain my ensemble architecture in scientific text suitable for IEEE format.`
- `Turn my evaluation scripts into a reproducibility appendix.`
- `Write a manuscript subsection from this repository without inventing anything.`

---

## Recommended Response Pattern
When invoked, the agent should usually respond in this order:
1. brief repository understanding,
2. verified methodological summary,
3. LaTeX-ready scientific text,
4. short list of unresolved gaps or placeholders.

---

## Optional Extension
If the user wants stronger traceability, also generate a companion artifact such as:
- `traceability.md`
- `method_sources.md`
- `paper_claims_checklist.md`

This companion file should map each paragraph to:
- source file,
- function/class names,
- config keys,
- confidence level.

---

## Success Criteria
This skill is successful when it produces scientific prose that is:
- faithful to the code,
- suitable for inclusion in a paper,
- reproducible,
- LaTeX-ready,
- explicit about uncertainty,
- free from invented claims.

