# NLP-04 Reproducibility Guide

## Project

Multi-Label Toxic Comment Classification with Explainability.

The research compares classical TF-IDF classifiers, DistilBERT
and RoBERTa on a shared leakage-aware Jigsaw data protocol.

## Final selected model

- Run ID: `S02_ROBERTA_FOCAL_FULL_S42`
- Base model: `FacebookAI/roberta-base`
- Base revision: `e2da8e2f811d1448a5b465c236feacd80ffbac7b`
- Training samples: 127,628
- Training seed: 42
- Maximum sequence length: 256
- Independent sigmoid outputs: 6
- Selection metric: validation Macro Average Precision

## Execution order

Run the notebooks in the following order:

1. `00_project_setup/`
2. `01_dataset_eda/`
3. `02_data_preparation/`
4. `03_classical_ml/`
5. `04_distilbert/`
6. `05_roberta/`
7. `06_model_optimization/`
8. `07_final_evaluation/`
9. `08_error_analysis/`
10. `09_fairness_audit/`
11. `10_explainability/`
12. `11_final_documentation/`

The classical ML stage contains multiple submodules.

Use GPU for transformer training and GPU-dependent inference.
Dataset analysis, metric consolidation and final documentation
can be performed on CPU.

## Data requirements

Obtain the Jigsaw Toxic Comment Classification Challenge dataset:

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

Required source data include train.csv, test.csv and test_labels.csv.

Use the original setup and data preparation notebooks to verify
raw files and regenerate the prepared partitions.

The prepared dataset manifest is:

`01_data/splits/prepared_dataset_manifest.csv`

The official scored test partition contains 63,978
comments with valid annotations for all six labels.

Training and validation data must remain separate from official test
data. Raw comments are preserved for modeling. Lowercase and
whitespace normalization are used only for duplicate detection.

## Environment and dependencies

Use the recorded experiment configurations and original notebook
environment setup rather than assuming the newest library versions
will reproduce identical results.

The GitHub publication includes observed dependency information
where available in requirements-observed.txt.

The public repository does not contain all datasets, model binaries,
training checkpoints or private evaluation artifacts.

Exact replay requires the original verified files or compatible
artifacts that pass the saved fingerprint checks.

## Frozen evaluation protocol

Do not tune thresholds or select models on the official test set.

The authoritative frozen manifest is:

`02_notebooks/06_model_optimization/results/configs/frozen_manifest.json`

SHA-256:

`229adb5d6166a626f6b7f7fddecf38b4de58d21e56ec4367465ad29174ef2fd1`

The final checkpoint directory is:

`04_models/optimization/S02_ROBERTA_FOCAL_FULL_S42/best_epoch_03_20261126_120001_402121`

The model files and full-precision thresholds are documented in
results/configs/final_model_manifest.json.

## Frozen decision thresholds

- `toxic`: 0.53506803512573242
- `severe_toxic`: 0.46651136875152588
- `obscene`: 0.54249972105026245
- `threat`: 0.48011302947998047
- `insult`: 0.48708629608154297
- `identity_hate`: 0.48673561215400696

Apply each threshold independently using score >= threshold.

Do not replace these values with rounded report values or
re-optimize using the test labels.

## Verified official test results

- Macro AP: 0.685670
- Macro ROC-AUC: 0.985466
- Macro F1: 0.617611
- Micro F1: 0.682986

All values are derived from saved evaluation artifacts.

## Research evidence

The final documentation directory contains:

- T01_final_comparison_index.csv
- T02_requirement_evidence_matrix.csv
- T03_final_thresholds_and_cost.csv
- report_ppt_asset_index.csv
- presentation_asset_index.csv
- research_limitations.csv
- deferred_experiments.csv
- deployment_brief_final.md
- final_model_manifest.json

The Cell 3 requirement matrix is a historical snapshot. Deployment
completion is recorded separately until the final verification cell.

Original figures and tables are the authoritative evidence.
Presentation copies preserve the same file contents and SHA-256 hashes.

## Explainability and fairness

The frozen final model has 20 LIME explanations and 460
faithfulness evaluation records.

The fairness audit includes natural identity-term comparisons
and controlled synthetic comments.

These analyses do not establish causal explanations, demographic
fairness or production safety.

Independent human validation remains pending for AI-assisted
qualitative interpretations and synthetic audit intent.

## Deferred experiments

The following experiments were deferred:

S03, R01, A01, A02, A03

These are not included as completed runs.

Any new post-freeze model changes must be identified as exploratory
unless evaluated using a fresh independent benchmark.

## Publication and privacy

Do not publish raw comments, raw-token explanation HTML,
individual explanation figures containing comment words,
private interpretation worksheets, or per-comment identifiers
and prediction records without separate review.

Exclude private_explanations, private_documentation and other
private review directories from the public repository.

Review notebook outputs and source cells for embedded private
content before publication.

Aggregate tables and figures should be reviewed before sharing.

## Deployment scope

The model is intended to support human moderation decisions.

It has not been validated as a production service. Offline
GPU throughput is not single-request latency.

The final deployment brief documents workload, fairness,
thresholds, limitations and proposed monitoring.
