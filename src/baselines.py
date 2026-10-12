from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import (
    ATTACK_CATEGORY_MAP,
    load_data,
    preprocess_data,
)
from src.evaluation import (
    accuracy_score,
    compute_metrics,
    confusion_matrix,
    macro_precision_recall_f1,
    stratified_train_test_split,
)


class DecisionTreeNode:
    def __init__(
        self,
        is_leaf: bool = False,
        pred: Optional[int] = None,
        probas: Optional[np.ndarray] = None,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["DecisionTreeNode"] = None,
        right: Optional["DecisionTreeNode"] = None,
        gain: float = 0.0,
    ) -> None:
        self.is_leaf = is_leaf
        self.pred = pred
        self.probas = probas
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain


class DecisionTreeClassifier:
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = 12,
        min_samples_split: int = 20,
        min_samples_leaf: int = 5,
        max_features: Optional[Union[int, float, str]] = None,
        n_bins: int = 32,
        random_state: Optional[int] = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_bins = n_bins
        self.random_state = random_state
        self.root: Optional[DecisionTreeNode] = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    def _impurity(self, counts: np.ndarray, n_samples: int) -> float:
        if n_samples == 0:
            return 0.0
        p = counts / n_samples
        if self.criterion == "entropy":
            p_pos = p[p > 0]
            return float(-np.sum(p_pos * np.log2(p_pos)))
        return float(1.0 - np.sum(p**2))

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> Tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0.0

        if self.max_features is None:
            feat_indices = np.arange(n_features)
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
            feat_indices = rng.choice(n_features, size=k, replace=False)
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
            feat_indices = rng.choice(n_features, size=k, replace=False)
        elif isinstance(self.max_features, int):
            feat_indices = rng.choice(
                n_features, size=min(self.max_features, n_features), replace=False
            )
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
            feat_indices = rng.choice(n_features, size=k, replace=False)
        else:
            feat_indices = np.arange(n_features)

        parent_counts = np.bincount(y, minlength=self.n_classes_)
        parent_impurity = self._impurity(parent_counts, n_samples)
        if parent_impurity == 0.0:
            return None, None, 0.0

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        y_onehot = np.zeros((n_samples, self.n_classes_), dtype=np.float32)
        y_onehot[np.arange(n_samples), y] = 1.0

        for feat in feat_indices:
            x_col = X[:, feat]
            order = np.argsort(x_col)
            x_sorted = x_col[order]
            y_sorted_oh = y_onehot[order]

            diff = np.diff(x_sorted)
            change_mask = diff > 1e-7
            if not np.any(change_mask):
                continue
            change_indices = np.where(change_mask)[0]

            if len(change_indices) > self.n_bins:
                sub_sel = np.linspace(0, len(change_indices) - 1, self.n_bins, dtype=int)
                cand_indices = change_indices[sub_sel]
            else:
                cand_indices = change_indices

            cum_counts = np.cumsum(y_sorted_oh, axis=0)
            left_counts = cum_counts[cand_indices]
            n_left = cand_indices + 1
            n_right = n_samples - n_left

            valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not np.any(valid):
                continue

            cand_indices = cand_indices[valid]
            left_counts = left_counts[valid]
            n_left = n_left[valid]
            n_right = n_right[valid]
            right_counts = parent_counts - left_counts

            if self.criterion == "gini":
                p_l = left_counts / n_left[:, None]
                p_r = right_counts / n_right[:, None]
                imp_l = 1.0 - np.sum(p_l**2, axis=1)
                imp_r = 1.0 - np.sum(p_r**2, axis=1)
            else:
                p_l = left_counts / n_left[:, None]
                p_r = right_counts / n_right[:, None]
                p_l = np.where(p_l > 0, p_l, 1.0)
                p_r = np.where(p_r > 0, p_r, 1.0)
                imp_l = -np.sum(p_l * np.log2(p_l), axis=1)
                imp_r = -np.sum(p_r * np.log2(p_r), axis=1)

            weighted_imp = (n_left / n_samples) * imp_l + (n_right / n_samples) * imp_r
            gains = parent_impurity - weighted_imp

            max_idx = np.argmax(gains)
            if gains[max_idx] > best_gain:
                best_gain = float(gains[max_idx])
                best_feat = int(feat)
                split_idx = cand_indices[max_idx]
                best_thresh = float((x_sorted[split_idx] + x_sorted[split_idx + 1]) / 2.0)

        return best_feat, best_thresh, best_gain

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator
    ) -> DecisionTreeNode:
        counts = np.bincount(y, minlength=self.n_classes_)
        pred = int(np.argmax(counts))
        probas = counts / len(y)

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
        ):
            return DecisionTreeNode(is_leaf=True, pred=pred, probas=probas)

        feat, thresh, gain = self._best_split(X, y, rng)
        if feat is None or gain <= 0.0:
            return DecisionTreeNode(is_leaf=True, pred=pred, probas=probas)

        if self.feature_importances_ is not None:
            self.feature_importances_[feat] += gain * len(y)

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng)

        return DecisionTreeNode(
            is_leaf=False,
            pred=pred,
            probas=probas,
            feature=feat,
            threshold=thresh,
            left=left_node,
            right=right_node,
            gain=gain,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = int(np.max(y)) + 1
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)

        rng = np.random.default_rng(self.random_state)
        self.root = self._build_tree(X, y, 0, rng)

        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp

        return self

    def _predict_row(self, node: DecisionTreeNode, x: np.ndarray) -> int:
        curr = node
        while not curr.is_leaf:
            if curr.feature is not None and curr.threshold is not None:
                if x[curr.feature] <= curr.threshold:
                    if curr.left is not None:
                        curr = curr.left
                    else:
                        break
                else:
                    if curr.right is not None:
                        curr = curr.right
                    else:
                        break
            else:
                break
        return curr.pred if curr.pred is not None else 0

    def _predict_proba_row(self, node: DecisionTreeNode, x: np.ndarray) -> np.ndarray:
        curr = node
        while not curr.is_leaf:
            if curr.feature is not None and curr.threshold is not None:
                if x[curr.feature] <= curr.threshold:
                    if curr.left is not None:
                        curr = curr.left
                    else:
                        break
                else:
                    if curr.right is not None:
                        curr = curr.right
                    else:
                        break
            else:
                break
        return (
            curr.probas
            if curr.probas is not None
            else np.zeros(self.n_classes_, dtype=np.float32)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.root is None:
            raise RuntimeError("DecisionTreeClassifier is not fitted yet.")
        return np.array([self._predict_row(self.root, x) for x in X], dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.root is None:
            raise RuntimeError("DecisionTreeClassifier is not fitted yet.")
        return np.array([self._predict_proba_row(self.root, x) for x in X], dtype=np.float32)


class KNeighborsClassifier:
    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "distance",
        batch_size: int = 1000,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.batch_size = batch_size
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighborsClassifier":
        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        self.n_classes_ = int(np.max(self.y_train_)) + 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("KNeighborsClassifier is not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        n_test = len(X)
        probas = np.zeros((n_test, self.n_classes_), dtype=np.float32)

        for i in range(0, n_test, self.batch_size):
            X_batch = X[i : i + self.batch_size]
            if self.metric == "euclidean":
                dists = (
                    np.sum(X_batch**2, axis=1, keepdims=True)
                    + np.sum(self.X_train_**2, axis=1, keepdims=True).T
                    - 2.0 * np.dot(X_batch, self.X_train_.T)
                )
                dists = np.maximum(dists, 0.0)
                dists = np.sqrt(dists)
            else:
                dists = np.sum(
                    np.abs(X_batch[:, None, :] - self.X_train_[None, :, :]), axis=2
                )

            k = min(self.n_neighbors, self.X_train_.shape[0])
            top_k_idx = np.argpartition(dists, k, axis=1)[:, :k]

            for row_idx in range(len(X_batch)):
                knn_labels = self.y_train_[top_k_idx[row_idx]]
                if self.weights == "distance":
                    knn_dists = dists[row_idx, top_k_idx[row_idx]]
                    w = 1.0 / (knn_dists + 1e-6)
                    for l, weight_val in zip(knn_labels, w):
                        probas[i + row_idx, l] += weight_val
                    sum_w = np.sum(probas[i + row_idx])
                    if sum_w > 0:
                        probas[i + row_idx] /= sum_w
                else:
                    counts = np.bincount(knn_labels, minlength=self.n_classes_)
                    probas[i + row_idx] = counts / k

        return probas


def evaluate_baseline(
    model: Union[DecisionTreeClassifier, KNeighborsClassifier],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t1

    metrics = compute_metrics(y_test, y_pred)
    metrics["fit_time"] = fit_time
    metrics["pred_time"] = pred_time
    metrics["predictions"] = y_pred

    return metrics


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "raw",
        "networkTraffic.csv",
    )

    raw_df = load_data(data_path).head(10000)
    X, y, feature_names, _ = preprocess_data(raw_df, is_train=True)
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.20, random_state=69
    )

    dt = DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=69)
    res = evaluate_baseline(dt, X_train, y_train, X_test, y_test)
    print(f"Decision Tree Acc: {res['accuracy']:.4f}, Macro F1: {res['macro_f1']:.4f}")

    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    res_knn = evaluate_baseline(knn, X_train[:2000], y_train[:2000], X_test[:500], y_test[:500])
    print(f"k-NN Acc: {res_knn['accuracy']:.4f}, Macro F1: {res_knn['macro_f1']:.4f}")