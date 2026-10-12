from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ATTACK_CATEGORY_MAP: Dict[int, str] = {
    0: "Normal",
    1: "Reconnaissance",
    2: "Backdoor",
    3: "DoS",
    4: "Exploits",
    5: "Analysis",
    6: "Fuzzers",
    7: "Worms",
    8: "Shellcode",
    9: "Generic",
}

ATTACK_NAME_TO_ID: Dict[str, int] = {v: k for k, v in ATTACK_CATEGORY_MAP.items()}

NOMINAL_COLUMNS: List[str] = ["proto", "service", "state"]
ID_COLUMN: str = "id"
TARGET_COLUMN: str = "attack_cat"


def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at path: {filepath}")
    return pd.read_csv(filepath)


class NetworkTrafficPreprocessor:
    def __init__(
        self,
        proto_threshold: Union[int, float] = 0.001,
        missing_service_label: str = "unknown",
        scale_features: bool = False,
        dtype: type = np.float32,
    ) -> None:
        self.proto_threshold = proto_threshold
        self.missing_service_label = missing_service_label
        self.scale_features = scale_features
        self.dtype = dtype

        self.frequent_protocols_: set[str] = set()
        self.feature_names_: list[str] = []
        self.numerical_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.category_maps_: dict[str, dict[str, int]] = {}
        self.num_means_: Optional[np.ndarray] = None
        self.num_stds_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def _clean_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()

        if ID_COLUMN in X_clean.columns:
            X_clean = X_clean.drop(columns=[ID_COLUMN])

        if TARGET_COLUMN in X_clean.columns:
            X_clean = X_clean.drop(columns=[TARGET_COLUMN])

        if "service" in X_clean.columns:
            X_clean["service"] = (
                X_clean["service"]
                .replace("?", self.missing_service_label)
                .fillna(self.missing_service_label)
                .astype(str)
            )

        return X_clean

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "NetworkTrafficPreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        X_clean = self._clean_dataframe(X)

        self.categorical_cols_ = [col for col in NOMINAL_COLUMNS if col in X_clean.columns]
        self.numerical_cols_ = [
            col for col in X_clean.columns if col not in self.categorical_cols_
        ]
        self.feature_names_ = list(X_clean.columns)

        if "proto" in X_clean.columns:
            proto_counts = X_clean["proto"].value_counts()
            n_samples = len(X_clean)
            if isinstance(self.proto_threshold, float) and self.proto_threshold < 1.0:
                min_count = self.proto_threshold * n_samples
            else:
                min_count = float(self.proto_threshold)
            self.frequent_protocols_ = set(proto_counts[proto_counts >= min_count].index)
        else:
            self.frequent_protocols_ = set()

        self.category_maps_ = {}
        for col in self.categorical_cols_:
            series = X_clean[col].copy()
            if col == "proto":
                series = series.apply(
                    lambda p: p if p in self.frequent_protocols_ else "other"
                )
            unique_vals = sorted(series.unique())
            self.category_maps_[col] = {val: idx for idx, val in enumerate(unique_vals)}

        if self.scale_features and len(self.numerical_cols_) > 0:
            num_data = X_clean[self.numerical_cols_].to_numpy(dtype=self.dtype)
            self.num_means_ = np.mean(num_data, axis=0)
            self.num_stds_ = np.std(num_data, axis=0)
            self.num_stds_[self.num_stds_ == 0.0] = 1.0
        else:
            self.num_means_ = None
            self.num_stds_ = None

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("NetworkTrafficPreprocessor must be fitted before calling transform.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        X_clean = self._clean_dataframe(X)

        for col in self.categorical_cols_:
            series = X_clean[col].copy()
            if col == "proto":
                series = series.apply(
                    lambda p: p if p in self.frequent_protocols_ else "other"
                )
            mapping = self.category_maps_[col]
            unknown_code = mapping.get("other", mapping.get(self.missing_service_label, -1))
            X_clean[col] = series.map(mapping).fillna(unknown_code).astype(self.dtype)

        if self.scale_features and self.num_means_ is not None and self.num_stds_ is not None:
            num_vals = X_clean[self.numerical_cols_].to_numpy(dtype=self.dtype)
            X_clean[self.numerical_cols_] = (num_vals - self.num_means_) / self.num_stds_

        return X_clean[self.feature_names_].to_numpy(dtype=self.dtype)

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> list[str]:
        if not self.is_fitted_:
            raise RuntimeError("NetworkTrafficPreprocessor is not fitted yet.")
        return list(self.feature_names_)


def preprocess_data(
    df: pd.DataFrame,
    is_train: bool = True,
    preprocessor: Optional[NetworkTrafficPreprocessor] = None,
    proto_threshold: Union[int, float] = 0.001,
    missing_service_label: str = "unknown",
    scale_features: bool = False,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    list[str],
    NetworkTrafficPreprocessor,
]:
    prep = preprocessor

    if is_train:
        if prep is None:
            prep = NetworkTrafficPreprocessor(
                proto_threshold=proto_threshold,
                missing_service_label=missing_service_label,
                scale_features=scale_features,
            )
        prep.fit(df)
    else:
        if prep is None:
            raise ValueError(
                "An existing fitted `preprocessor` must be provided when `is_train=False`."
            )

    X = prep.transform(df)
    feature_names = prep.get_feature_names_out()

    y: Optional[np.ndarray] = None
    if TARGET_COLUMN in df.columns:
        target_series = df[TARGET_COLUMN]
        if target_series.dtype == object:
            y = (
                target_series.map(ATTACK_NAME_TO_ID)
                .fillna(-1)
                .to_numpy(dtype=np.int64)
            )
        else:
            y = target_series.to_numpy(dtype=np.int64)

    return X, y, feature_names, prep


if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "raw",
        "networkTraffic.csv",
    )

    print(f"Loading raw dataset from: {data_path}")
    raw_df = load_data(data_path)
    print(f"Loaded {raw_df.shape[0]:,} rows and {raw_df.shape[1]} columns")

    X, y, feat_names, preprocessor = preprocess_data(
        raw_df,
        is_train=True,
        proto_threshold=0.001,
        missing_service_label="unknown",
        scale_features=False,
    )

    print(f"X shape: {X.shape} (dtype: {X.dtype})")
    print(f"y shape: {y.shape} (dtype: {y.dtype})")
    print(f"Total features: {len(feat_names)}")
    print(f"Numerical features: {len(preprocessor.numerical_cols_)}")
    print(f"Categorical features: {len(preprocessor.categorical_cols_)}")
    print(f"NaNs in X: {np.isnan(X).sum()}")
    print("Preprocessing completed successfully.")