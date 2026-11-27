# NLP-04: Deployment Brief

## 1. Intended use and model

The proposed system assists human moderators in identifying potentially
toxic English-language comments. It is not an automatic removal or
punishment system.

The selected model is RoBERTa with focal loss
(`S02_ROBERTA_FOCAL_FULL_S42`), fine-tuned on
127,628 Jigsaw training comments.
Model selection used validation Macro Average Precision.

Raw text is preserved and tokenized using the frozen RoBERTa tokenizer,
with right truncation at 256 tokens
and dynamic padding to a multiple of eight. Six independent sigmoid
scores are produced, allowing multiple toxicity labels per comment.

## 2. Verified performance and thresholds

Evaluation used 63,978 labelled official Jigsaw test comments.

- Macro AP: 0.6857
- Macro AP 95% bootstrap interval:
  [0.6699, 0.7026]
- Macro ROC-AUC: 0.9855
- Macro F1: 0.6176
- Micro F1: 0.6830

The following decision thresholds were optimized exclusively on
validation data and frozen before official test evaluation.

| Label | Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| toxic | 0.535068 | 0.5684 | 0.8949 | 0.6952 |
| severe_toxic | 0.466511 | 0.2780 | 0.6703 | 0.3930 |
| obscene | 0.542500 | 0.6245 | 0.8057 | 0.7037 |
| threat | 0.480113 | 0.4815 | 0.7393 | 0.5832 |
| insult | 0.487086 | 0.6182 | 0.8074 | 0.7002 |
| identity_hate | 0.486736 | 0.5578 | 0.7247 | 0.6304 |

A label is flagged when its score is greater than or equal to its
threshold. The full-precision thresholds in the model manifest, rather
than the rounded values above, must be used for implementation.

## 3. Proposed moderation workflow

1. Validate incoming comment text and reject invalid input safely.
2. Tokenize using the frozen preprocessing configuration.
3. Generate six scores and apply the corresponding thresholds.
4. Route comments with at least one positive flag to human moderation.
5. Provide optional LIME explanations for diagnostic review.
6. Let moderators consider context, quotation, counterspeech and
   identity-related language before making decisions.
7. Record decisions securely and monitor errors, workload and drift.

Unflagged comments are not guaranteed harmless and should remain
eligible for user reports and other safety checks.

## 4. Expected benchmark moderation workload

Per 1,000 observed official test comments, the optimized thresholds
would flag approximately 152.36 unique comments, including
65.18 comments annotated negative across all six labels.
Approximately 10.39 comments annotated positive for at least
one label would be missed entirely.

Default thresholds flag 164.15 comments and miss
8.39 positive comments per 1,000.

Threshold optimization reduces workload but increases missed positive
comments at the comment level. These are benchmark observations,
not guaranteed production rates.

## 5. Computational requirements

The frozen model weights occupy 475.53 MiB. The four required model and tokenizer files together occupy 478.92 MiB, excluding the separate checkpoint manifest and application overhead.

Offline batched GPU inference processed
309.98
comments per second with batch size
32.

Peak allocated GPU memory was
0.675 GiB during the
recorded benchmark.

Amortized inference time was
3.226
milliseconds per comment. This excludes unmeasured service overhead
and must not be interpreted as single-request API latency.

Production latency, total application memory, CPU serving performance
and concurrent-request capacity require separate benchmarking.

## 6. Fairness and explainability

On the natural test audit, the toxic-label false-positive rate was
11.63% for comments containing monitored identity terms,
compared with 6.82% for comments without those terms.
The observed gap was 4.81 percentage points.

The audit covered 30 identity terms
and 744 synthetic comments.
Residual identity sensitivity requires contextual human review.

LIME explanations were generated for
20 diagnostic cases.
Mean local surrogate R2 was
0.8398.
Top-feature removal produced larger absolute score changes than
matched random removal in
20
of those cases.

LIME is diagnostic rather than causal. Written interpretations are
AI-assisted and independent human validation remains pending.

## 7. Limitations and deployment safeguards

The training domain is English Wikipedia comments. Annotation noise,
rare labels, truncation, identity sensitivity and distribution shift
limit generalization. Sigmoid scores are not established as calibrated
probabilities. Bootstrap intervals do not capture training-seed
variability.

Before operational use, validate on representative platform-specific
data, measure serving latency, review fairness and define moderator
escalation policies. Monitor flag rates, false positives, false
negatives and identity-term disparities. Restrict access to private
comments and maintain an appropriate data-retention policy.

Any future model or threshold changes require separate development
and fresh evaluation evidence. No production system has been deployed
or validated in this study.
