---
name: scientific-validation-review
description: >
  Use when reviewing research or experimental code to detect scientific,
  methodological, statistical, or data-handling flaws that could invalidate
  results, bias metrics, or undermine reproducibility.
compatibility: opencode
---

# Scientific Validation Review for Research Code

This skill enables the agent to review code that supports scientific
experiments, model training, data analysis, or benchmarking pipelines.

The objective is to detect **scientific flaws that could invalidate results**
even if the code is technically correct.

This skill goes beyond software quality checks and must explicitly
analyze the **scientific validity of the implementation and experimental design**.

Use this skill when evaluating research code that may produce
metrics, claims, or discoveries.

---

# Primary Objective

Identify any **implementation, methodological, statistical, or data-handling
issue** that could make conclusions:

- unreliable
- biased
- irreproducible
- inflated
- statistically invalid
- scientifically indefensible

Focus on threats to:

- internal validity
- external validity
- reproducibility
- statistical validity
- evaluation fairness
- scientific defensibility

---

# When to Use

Activate this skill when reviewing code related to:

- machine learning training or evaluation
- dataset preprocessing for experiments
- dataset splitting logic
- statistical analysis pipelines
- metric computation
- hyperparameter optimization
- threshold tuning
- ablation studies
- benchmarking systems
- inference pipelines used in scientific claims
- medical, biological, physical, or other scientific research code
- code supporting academic publications or reports

---

# Core Scientific Review Principles

Always evaluate code according to the following principles:

1. **Reproducibility**
2. **Data leakage prevention**
3. **Correct statistical methodology**
4. **Fair evaluation procedures**
5. **Transparent experimental design**
6. **Metric validity**
7. **Scientific defensibility**

A system may be technically correct but still scientifically invalid.

---

# Scientific Review Workflow

When reviewing research code, the agent must follow this process:

1. Understand the scientific objective of the code
2. Identify datasets and experimental boundaries
3. Inspect preprocessing steps
4. Analyze dataset splitting logic
5. Verify training and evaluation separation
6. Inspect metric computation
7. Evaluate statistical methodology
8. Detect leakage or bias risks
9. Assess reproducibility guarantees
10. Report all threats to validity

---

# Critical Scientific Checks

## Dataset Splitting Integrity

Verify that:

- train, validation, and test sets are strictly separated
- no sample leakage occurs
- no derived data from test appears in training
- group-based or patient-based splitting is respected when required

Common failure cases:

- random splitting where group-level separation is required
- augmentations leaking information across splits
- normalization computed using full dataset

---

## Data Leakage Detection

Check for leakage sources such as:

- preprocessing fitted on the entire dataset
- normalization using global statistics
- feature engineering derived from target variables
- hyperparameter tuning using the test set
- implicit leakage through cached artifacts

Any leakage invalidates reported metrics.

---

## Metric Validity

Confirm that metrics are:

- computed on the correct dataset partition
- implemented correctly
- not inflated by preprocessing artifacts
- appropriate for the task

Examples:

- using accuracy for highly imbalanced datasets
- incorrect IoU or Dice implementations
- computing metrics before thresholding
- averaging metrics incorrectly

---

## Statistical Validity

Inspect for statistical flaws such as:

- insufficient sample size
- lack of confidence intervals
- absence of repeated runs
- p-hacking through repeated experimentation
- cherry-picked hyperparameters

When possible, recommend:

- bootstrap confidence intervals
- cross-validation
- statistical significance testing

---

## Evaluation Fairness

Verify that evaluation procedures are fair:

- identical preprocessing for all methods
- same dataset splits across experiments
- identical evaluation metrics
- identical inference conditions

Benchmark comparisons must be scientifically fair.

---

## Reproducibility Checks

Ensure experiments can be reproduced:

- deterministic seeds are set
- random number generators are controlled
- training configurations are logged
- dataset versions are fixed
- environment dependencies are documented

Without reproducibility, claims cannot be validated.

---

# Common Scientific Failure Patterns

The agent must explicitly detect patterns such as:

- dataset leakage
- metric inflation
- incorrect baseline comparisons
- hidden preprocessing bias
- evaluation on training data
- hyperparameter tuning on test data
- selective reporting of results

These issues invalidate scientific conclusions.

---

# Expected Output

The agent should produce a structured scientific review including:

1. **Summary of scientific objective**
2. **Detected threats to validity**
3. **Severity classification**

Severity levels:

- **Critical** — invalidates conclusions
- **Major** — significantly biases results
- **Moderate** — reduces reliability
- **Minor** — small methodological weakness

4. **Recommended corrections**
5. **Reproducibility improvements**

---

# Key Rule

If a flaw could invalidate scientific claims,
it must be clearly highlighted even if the code executes correctly.

Correct software is not necessarily correct science.
