# NLP-04 Final Research Overview

## Selected model

RoBERTa with focal loss: `S02_ROBERTA_FOCAL_FULL_S42`.

The model was selected using validation Macro Average Precision.
It was trained on 127,628 comments.

A total of 19 model configurations
were evaluated using the same frozen official test protocol.

## Final performance

| Metric | Validation | Official test |
|---|---:|---:|
| Macro Average Precision | 0.753858 | 0.685670 |
| Macro F1 | 0.712581 | 0.617611 |
| Macro ROC-AUC | See validation evidence | 0.985466 |
| Micro F1 | See validation evidence | 0.682986 |

Official labelled test population: 63,978.

The selected model achieved the highest Macro AP among the
completed configurations. This does not establish superiority
over all existing approaches or state-of-the-art models.

## Explainability and fairness

- Error-analysis cases: 180
- Identity terms audited: 30
- Synthetic fairness comments: 744
- LIME explanations: 20
- Mean local surrogate R2: 0.839804
- Faithfulness evaluations: 460
- Top deletion exceeds random mean absolute change:
  20 cases
- Retained prediction agreement:
  85.00%

Residual identity-term sensitivity remains. LIME explanations
are local diagnostic approximations rather than causal evidence.

## Proposed moderation workload

Per 1,000 observed benchmark comments:

- Flagged comments: 152.36
- Incorrectly flagged all-label negative comments:
  65.18
- Missed positive comments:
  10.39

These are benchmark observations, not production forecasts.

The deployment proposal uses human-assisted moderation.
Production deployment has not been performed.

## Requirement completion

All nine core requirements documented in the agreed master
execution plan are completed.

The original university brief was not independently reverified
during the final documentation stage.

Independent human review of the AI-assisted qualitative
interpretations and synthetic fairness intent remains pending.

## Deferred experiments

S03, R01, A01, A02, A03

Deferred experiments are excluded from completed results.

## Remaining submission activities

- Independent human validation.
- Final academic report and PPT assembly.
- Public GitHub repository update with privacy-safe artifacts.
- Complete verified project backup.

## Main evidence

- T01_final_comparison_index.csv
- T02_requirement_evidence_matrix_final.csv
- T03_final_thresholds_and_cost.csv
- report_ppt_asset_index.csv
- research_limitations.csv
- deployment_brief_final.md
- reproducibility_README.md
- final_model_manifest.json

All reported numerical results are derived from verified saved
experimental evidence.
