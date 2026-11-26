
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    f1_score,
    hamming_loss,
    accuracy_score
)


def validate_inputs(y_true, y_scores, labels):

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    if y_true.ndim != 2 or y_true.shape != y_scores.shape:
        raise ValueError("Invalid target or score dimensions")

    if y_true.shape[1] != len(labels):
        raise ValueError("Label count does not match input dimensions")

    if len(set(labels)) != len(labels):
        raise ValueError("Label names must be unique")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Targets must contain binary labels")

    if not np.isfinite(y_scores).all():
        raise ValueError("Prediction scores contain invalid values")

    if any(np.unique(y_true[:, i]).size != 2
           for i in range(len(labels))):
        raise ValueError("Each label must contain both classes")

    return y_true.astype(np.int8), y_scores


def apply_thresholds(y_scores, thresholds):

    y_scores = np.asarray(y_scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    if y_scores.ndim != 2:
        raise ValueError("Prediction scores must be two-dimensional")

    if thresholds.ndim == 0:
        thresholds = np.full(y_scores.shape[1], thresholds.item())

    if thresholds.shape != (y_scores.shape[1],):
        raise ValueError("Invalid threshold dimensions")

    if not np.isfinite(thresholds).all():
        raise ValueError("Thresholds contain invalid values")

    return (y_scores >= thresholds).astype(np.int8)


def evaluate_multilabel(y_true, y_scores, labels, thresholds=0.5):

    y_true, y_scores = validate_inputs(
        y_true, y_scores, labels
    )

    y_pred = apply_thresholds(y_scores, thresholds)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0
        )
    )

    ap_scores = average_precision_score(
        y_true, y_scores, average=None
    )

    auc_scores = roc_auc_score(
        y_true, y_scores, average=None
    )

    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0)
    tn = ((y_true == 0) & (y_pred == 0)).sum(axis=0)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0)

    threshold_values = np.asarray(thresholds)

    if threshold_values.ndim == 0:
        threshold_values = np.full(
            len(labels), threshold_values.item()
        )

    per_label = pd.DataFrame({
        "Label": labels,
        "Threshold": threshold_values,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Average Precision": ap_scores,
        "ROC-AUC": auc_scores,
        "Positive Support": support,
        "Negative Support": len(y_true) - support,
        "Predicted Positives": y_pred.sum(axis=0),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    })

    overall = {
        "Macro AP": float(np.mean(ap_scores)),
        "Micro AP": float(average_precision_score(
            y_true, y_scores, average="micro"
        )),
        "Macro ROC-AUC": float(np.mean(auc_scores)),
        "Micro ROC-AUC": float(roc_auc_score(
            y_true, y_scores, average="micro"
        )),
        "Macro Precision": float(np.mean(precision)),
        "Macro Recall": float(np.mean(recall)),
        "Macro F1": float(np.mean(f1)),
        "Micro F1": float(f1_score(
            y_true, y_pred, average="micro", zero_division=0
        )),
        "Hamming Loss": float(hamming_loss(y_true, y_pred)),
        "Exact Match Accuracy": float(
            accuracy_score(y_true, y_pred)
        )
    }

    return overall, per_label


def optimize_thresholds(y_true, y_scores, labels):

    y_true, y_scores = validate_inputs(
        y_true, y_scores, labels
    )

    records = []

    for i, label in enumerate(labels):

        precision, recall, thresholds = precision_recall_curve(
            y_true[:, i],
            y_scores[:, i]
        )

        # The final precision-recall point has no threshold
        precision = precision[:-1]
        recall = recall[:-1]

        denominator = precision + recall

        f1 = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0
        )

        # Prefer the highest threshold when F1 scores are tied
        best_indices = np.flatnonzero(
            np.isclose(
                f1, f1.max(),
                rtol=0, atol=1e-12
            )
        )

        best_idx = best_indices[-1]

        records.append({
            "Label": label,
            "Optimal Threshold": float(thresholds[best_idx]),
            "Validation Precision": float(precision[best_idx]),
            "Validation Recall": float(recall[best_idx]),
            "Validation F1": float(f1[best_idx])
        })

    return pd.DataFrame(records)
