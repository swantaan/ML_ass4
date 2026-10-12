from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import ATTACK_CATEGORY_MAP


class StratifiedKFold:
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(y)
        rng = np.random.default_rng(self.random_state)
        classes = np.unique(y)

        fold_buckets: list[list[int]] = [[] for _ in range(self.n_splits)]

        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            if self.shuffle:
                rng.shuffle(cls_idx)
            splits = np.array_split(cls_idx, self.n_splits)
            for fold_i, split_idx in enumerate(splits):
                fold_buckets[fold_i].extend(split_idx.tolist())

        all_indices = np.arange(n_samples)
        for fold_i in range(self.n_splits):
            val_idx = np.array(fold_buckets[fold_i], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(val_idx)
            val_set = set(val_idx.tolist())
            train_idx = np.array(
                [idx for idx in all_indices if idx not in val_set], dtype=np.int64
            )
            if self.shuffle:
                rng.shuffle(train_idx)
            yield train_idx, val_idx


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    train_indices: list[int] = []
    test_indices: list[int] = []

    unique_classes = np.unique(y)
    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(np.round(len(cls_idx) * test_size))
        test_indices.extend(cls_idx[:n_test])
        train_indices.extend(cls_idx[n_test:])

    train_idx_arr = np.array(train_indices, dtype=np.int64)
    test_idx_arr = np.array(test_indices, dtype=np.int64)
    rng.shuffle(train_idx_arr)
    rng.shuffle(test_idx_arr)

    return X[train_idx_arr], X[test_idx_arr], y[train_idx_arr], y[test_idx_arr]


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None
) -> np.ndarray:
    if n_classes is None:
        n_classes = int(max(np.max(y_true), np.max(y_pred))) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None
) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    n = cm.shape[0]
    total_samples = len(y_true)

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []

    for i in range(n):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = int(np.sum(cm[i, :]))

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2.0 * p * r) / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

    supports_arr = np.array(supports, dtype=np.float64)
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    if total_samples > 0:
        weighted_precision = float(np.sum(np.array(precisions) * supports_arr) / total_samples)
        weighted_recall = float(np.sum(np.array(recalls) * supports_arr) / total_samples)
        weighted_f1 = float(np.sum(np.array(f1s) * supports_arr) / total_samples)
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
        "per_class_support": supports,
        "confusion_matrix": cm,
    }


def macro_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None
) -> Tuple[float, float, float]:
    m = compute_metrics(y_true, y_pred, n_classes=n_classes)
    return m["macro_precision"], m["macro_recall"], m["macro_f1"]


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[Dict[int, str]] = None,
    n_classes: Optional[int] = None,
) -> str:
    metrics = compute_metrics(y_true, y_pred, n_classes=n_classes)
    precisions = metrics["per_class_precision"]
    recalls = metrics["per_class_recall"]
    f1s = metrics["per_class_f1"]
    supports = metrics["per_class_support"]
    total_samples = len(y_true)

    lines = []
    header = f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}"
    lines.append(header)
    lines.append("-" * len(header))

    for i in range(len(precisions)):
        label = (
            target_names.get(i, f"Class {i}")
            if target_names is not None
            else f"Class {i}"
        )
        lines.append(
            f"{label:<20} {precisions[i]:<10.4f} {recalls[i]:<10.4f} {f1s[i]:<10.4f} {supports[i]:<10,}"
        )

    lines.append("-" * len(header))
    lines.append(
        f"{'Accuracy':<20} {'':<10} {'':<10} {metrics['accuracy']:<10.4f} {total_samples:<10,}"
    )
    lines.append(
        f"{'Macro Avg':<20} {metrics['macro_precision']:<10.4f} {metrics['macro_recall']:<10.4f} {metrics['macro_f1']:<10.4f} {total_samples:<10,}"
    )
    lines.append(
        f"{'Weighted Avg':<20} {metrics['weighted_precision']:<10.4f} {metrics['weighted_recall']:<10.4f} {metrics['weighted_f1']:<10.4f} {total_samples:<10,}"
    )

    return "\n".join(lines)


def evaluate_oob(
    oob_probas: np.ndarray, y_true: np.ndarray
) -> Dict[str, float]:
    valid_mask = np.sum(oob_probas, axis=1) > 0.0
    if not np.any(valid_mask):
        return {"oob_accuracy": 0.0, "oob_macro_f1": 0.0, "coverage": 0.0}

    y_valid = y_true[valid_mask]
    preds_valid = np.argmax(oob_probas[valid_mask], axis=1)

    acc = accuracy_score(y_valid, preds_valid)
    metrics = compute_metrics(y_valid, preds_valid, n_classes=oob_probas.shape[1])
    coverage = float(np.mean(valid_mask))

    return {
        "oob_accuracy": acc,
        "oob_macro_f1": metrics["macro_f1"],
        "coverage": coverage,
    }


def cross_validate(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: Optional[int] = 42,
) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_accuracies: list[float] = []
    fold_macro_f1s: list[float] = []
    fold_macro_precisions: list[float] = []
    fold_macro_recalls: list[float] = []
    fold_fit_times: list[float] = []
    fold_pred_times: list[float] = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = model_factory()

        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_t = time.time() - t0

        t1 = time.time()
        preds = model.predict(X_val)
        pred_t = time.time() - t1

        metrics = compute_metrics(y_val, preds)

        fold_accuracies.append(metrics["accuracy"])
        fold_macro_f1s.append(metrics["macro_f1"])
        fold_macro_precisions.append(metrics["macro_precision"])
        fold_macro_recalls.append(metrics["macro_recall"])
        fold_fit_times.append(fit_t)
        fold_pred_times.append(pred_t)

    return {
        "n_splits": n_splits,
        "fold_accuracies": fold_accuracies,
        "fold_macro_f1s": fold_macro_f1s,
        "fold_macro_precisions": fold_macro_precisions,
        "fold_macro_recalls": fold_macro_recalls,
        "fold_fit_times": fold_fit_times,
        "fold_pred_times": fold_pred_times,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_macro_f1": float(np.mean(fold_macro_f1s)),
        "std_macro_f1": float(np.std(fold_macro_f1s)),
        "mean_macro_precision": float(np.mean(fold_macro_precisions)),
        "std_macro_precision": float(np.std(fold_macro_precisions)),
        "mean_macro_recall": float(np.mean(fold_macro_recalls)),
        "std_macro_recall": float(np.std(fold_macro_recalls)),
        "mean_fit_time": float(np.mean(fold_fit_times)),
        "mean_pred_time": float(np.mean(fold_pred_times)),
    }


def cv_summary_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    records = [
        {"Metric": "Accuracy", "Mean": results["mean_accuracy"], "Std": results["std_accuracy"]},
        {"Metric": "Macro F1-Score", "Mean": results["mean_macro_f1"], "Std": results["std_macro_f1"]},
        {"Metric": "Macro Precision", "Mean": results["mean_macro_precision"], "Std": results["std_macro_precision"]},
        {"Metric": "Macro Recall", "Mean": results["mean_macro_recall"], "Std": results["std_macro_recall"]},
        {"Metric": "Fit Time (s)", "Mean": results["mean_fit_time"], "Std": float(np.std(results["fold_fit_times"]))},
        {"Metric": "Inference Time (s)", "Mean": results["mean_pred_time"], "Std": float(np.std(results["fold_pred_times"]))},
    ]
    return pd.DataFrame(records)


if __name__ == "__main__":
    from src.baselines import DecisionTreeClassifier
    from src.data_preprocessing import load_data, preprocess_data

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "raw",
        "networkTraffic.csv",
    )

    raw_df = load_data(data_path).head(10000)
    X, y, feature_names, _ = preprocess_data(raw_df, is_train=True)
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    dt = DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=ATTACK_CATEGORY_MAP))