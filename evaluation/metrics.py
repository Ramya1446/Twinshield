"""Consistent metric computation for every model on the same held-out test set."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from preprocessing.loader import CLASS_NAMES


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             class_names: Optional[List[str]] = None) -> Dict[str, float]:
    names = class_names or CLASS_NAMES
    labels = list(range(len(names)))
    kw = dict(labels=labels, zero_division=0)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision (macro)": float(precision_score(y_true, y_pred, average="macro", **kw)),
        "Recall (macro)": float(recall_score(y_true, y_pred, average="macro", **kw)),
        "Macro F1": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "Weighted F1": float(f1_score(y_true, y_pred, average="weighted", **kw)),
    }


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray,
                 class_names: Optional[List[str]] = None) -> Dict[str, float]:
    names = class_names or CLASS_NAMES
    scores = f1_score(y_true, y_pred, average=None,
                      labels=list(range(len(names))), zero_division=0)
    return {names[i]: float(scores[i]) for i in range(len(names))}


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: Optional[List[str]] = None) -> pd.DataFrame:
    names = class_names or CLASS_NAMES
    rep = classification_report(
        y_true, y_pred, labels=list(range(len(names))),
        target_names=names, output_dict=True, zero_division=0,
    )
    rows = []
    for n in names:
        r = rep[n]
        rows.append({
            "Class": n,
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1": round(r["f1-score"], 4),
            "Support": int(r["support"]),
        })
    return pd.DataFrame(rows)


def confusion(y_true: np.ndarray, y_pred: np.ndarray,
              class_names: Optional[List[str]] = None) -> np.ndarray:
    names = class_names or CLASS_NAMES
    return confusion_matrix(y_true, y_pred, labels=list(range(len(names))))


def comparison_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """results: {model_name: metrics dict} -> tidy DataFrame ordered as given."""
    rows = []
    for name, m in results.items():
        row = {"Model": name}
        row.update({k: round(v, 4) for k, v in m.items()})
        rows.append(row)
    return pd.DataFrame(rows)
