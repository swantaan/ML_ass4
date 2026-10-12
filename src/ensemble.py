from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.baselines import DecisionTreeClassifier
from src.data_preprocessing import (
    ATTACK_CATEGORY_MAP,
    load_data,
    preprocess_data,
)
from src.evaluation import (
    accuracy_score,
    classification_report,
    compute_metrics,
    stratified_train_test_split,
)


class BaggingClassifier:
    def __init__(
        self,
        estimator_cls: Any = DecisionTreeClassifier,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
        n_estimators: int = 25,
        max_samples: Union[int, float] = 1.0,
        bootstrap: bool = True,
        oob_score: bool = True,
        n_jobs: int = -1,
        random_state: Optional[int] = 69,
    ) -> None:
        self.estimator_cls = estimator_cls
        self.estimator_kwargs = estimator_kwargs or {}
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs if n_jobs > 0 else (os.cpu_count() or 4)
        self.random_state = random_state

        self.estimators_: list[Any] = []
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.oob_score_: float = 0.0
        self.oob_decision_function_: Optional[np.ndarray] = None

    def _fit_single_estimator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
        sample_size: int,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_samples = len(X)

        if self.bootstrap:
            boot_idx = rng.choice(n_samples, size=sample_size, replace=True)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False
            oob_idx = np.where(oob_mask)[0]
        else:
            boot_idx = rng.choice(n_samples, size=sample_size, replace=False)
            oob_idx = np.array([], dtype=np.int64)

        kwargs = dict(self.estimator_kwargs)
        if "random_state" in kwargs or hasattr(self.estimator_cls, "random_state"):
            kwargs["random_state"] = seed

        est = self.estimator_cls(**kwargs)
        est.fit(X[boot_idx], y[boot_idx])
        return est, boot_idx, oob_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self.n_classes_ = int(np.max(y)) + 1

        if isinstance(self.max_samples, float) and self.max_samples <= 1.0:
            sample_size = max(1, int(self.max_samples * n_samples))
        else:
            sample_size = int(self.max_samples)

        base_rng = np.random.default_rng(self.random_state)
        seeds = base_rng.integers(0, 1000000, size=self.n_estimators)

        tasks = [(X, y, int(seeds[i]), sample_size) for i in range(self.n_estimators)]

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(
                executor.map(
                    lambda p: self._fit_single_estimator(*p),
                    tasks,
                )
            )

        self.estimators_ = [r[0] for r in results]

        if self.oob_score and self.bootstrap:
            oob_probas = np.zeros((n_samples, self.n_classes_), dtype=np.float32)
            oob_counts = np.zeros(n_samples, dtype=np.int32)

            for est, _, oob_idx in results:
                if len(oob_idx) > 0:
                    tree_oob_preds = est.predict_proba(X[oob_idx])
                    oob_probas[oob_idx] += tree_oob_preds
                    oob_counts[oob_idx] += 1

            valid = oob_counts > 0
            if np.any(valid):
                oob_probas[valid] /= oob_counts[valid, None]
                oob_preds = np.argmax(oob_probas[valid], axis=1)
                self.oob_score_ = float(accuracy_score(y[valid], oob_preds))
                self.oob_decision_function_ = oob_probas
            else:
                self.oob_score_ = 0.0
                self.oob_decision_function_ = None

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if not self.estimators_:
            raise RuntimeError("BaggingClassifier is not fitted yet.")

        n_samples = len(X)
        accumulated_probas = np.zeros((n_samples, self.n_classes_), dtype=np.float32)

        for est in self.estimators_:
            accumulated_probas += est.predict_proba(X)

        accumulated_probas /= len(self.estimators_)
        return accumulated_probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1).astype(np.int64)


class RandomForestClassifier(BaggingClassifier):
    def __init__(
        self,
        n_estimators: int = 50,
        criterion: str = "gini",
        max_depth: Optional[int] = 12,
        min_samples_split: int = 15,
        min_samples_leaf: int = 2,
        max_features: Union[str, int, float, None] = "sqrt",
        n_bins: int = 32,
        bootstrap: bool = True,
        max_samples: Union[int, float] = 1.0,
        oob_score: bool = True,
        n_jobs: int = -1,
        random_state: Optional[int] = 69,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_bins = n_bins

        estimator_kwargs = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "n_bins": n_bins,
        }

        super().__init__(
            estimator_cls=DecisionTreeClassifier,
            estimator_kwargs=estimator_kwargs,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        super().fit(X, y)

        total_importances = np.zeros(self.n_features_in_, dtype=np.float64)
        for tree in self.estimators_:
            if hasattr(tree, "feature_importances_") and tree.feature_importances_ is not None:
                total_importances += tree.feature_importances_

        imp_sum = np.sum(total_importances)
        if imp_sum > 0:
            self.feature_importances_ = total_importances / imp_sum
        else:
            self.feature_importances_ = total_importances

        return self


def get_rf_configurations() -> Dict[str, Dict[str, Any]]:
    return {
        "baseline_rf": {
            "n_estimators": 25,
            "max_features": "sqrt",
            "max_depth": 12,
            "min_samples_split": 20,
            "criterion": "gini",
        },
        "deep_rf": {
            "n_estimators": 50,
            "max_features": "sqrt",
            "max_depth": 16,
            "min_samples_split": 10,
            "criterion": "gini",
        },
        "log2_subspace_rf": {
            "n_estimators": 50,
            "max_features": "log2",
            "max_depth": 14,
            "min_samples_split": 15,
            "criterion": "gini",
        },
        "entropy_rf": {
            "n_estimators": 50,
            "max_features": "sqrt",
            "max_depth": 14,
            "min_samples_split": 15,
            "criterion": "entropy",
        },
        "large_ensemble_rf": {
            "n_estimators": 100,
            "max_features": "sqrt",
            "max_depth": 16,
            "min_samples_split": 10,
            "criterion": "gini",
        },
    }


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

    rf = RandomForestClassifier(
        n_estimators=10,
        max_features="sqrt",
        max_depth=8,
        min_samples_split=20,
        oob_score=True,
        random_state=69,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(f"OOB Accuracy: {rf.oob_score_:.4f}")
    print(classification_report(y_test, y_pred, target_names=ATTACK_CATEGORY_MAP))