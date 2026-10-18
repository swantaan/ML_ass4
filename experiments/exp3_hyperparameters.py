from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src


def run_hyperparameter_study(quick: bool = False) -> pd.DataFrame:
    data_path = os.path.join(project_root, "data", "raw", "networkTraffic.csv")
    fig_dir = os.path.join(project_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 75)
    print("Experiment 3: Random Forest Hyperparameter & Subspace Study")
    print("=" * 75)

    raw_df = src.load_data(data_path)
    if quick:
        raw_df = raw_df.head(20000)
        n_est = 15
    else:
        raw_df = raw_df.head(50000)
        n_est = 25

    X, y, feature_names, _ = src.preprocess_data(raw_df, is_train=True)
    X_train, X_test, y_train, y_test = src.stratified_train_test_split(
        X, y, test_size=0.20, random_state=69
    )

    configs = [
        {"name": "Subspace: log2(D)", "max_features": "log2", "max_depth": 12, "criterion": "gini"},
        {"name": "Subspace: sqrt(D) [Default]", "max_features": "sqrt", "max_depth": 12, "criterion": "gini"},
        {"name": "Subspace: 0.5 * D", "max_features": 0.5, "max_depth": 12, "criterion": "gini"},
        {"name": "Subspace: All D (Bagging)", "max_features": None, "max_depth": 12, "criterion": "gini"},
        {"name": "Depth: Shallow (d=8)", "max_features": "sqrt", "max_depth": 8, "criterion": "gini"},
        {"name": "Depth: Deep (d=16)", "max_features": "sqrt", "max_depth": 16, "criterion": "gini"},
        {"name": "Criterion: Entropy", "max_features": "sqrt", "max_depth": 12, "criterion": "entropy"},
    ]

    records = []

    for cfg in configs:
        print(f"\nEvaluating configuration: {cfg['name']}...")
        rf = src.RandomForestClassifier(
            n_estimators=n_est,
            max_features=cfg["max_features"],
            max_depth=cfg["max_depth"],
            min_samples_split=20,
            criterion=cfg["criterion"],
            oob_score=True,
            random_state=69,
        )

        t0 = time.time()
        rf.fit(X_train, y_train)
        fit_time = time.time() - t0

        t1 = time.time()
        y_pred = rf.predict(X_test)
        pred_time = time.time() - t1

        metrics = src.compute_metrics(y_test, y_pred, n_classes=10)

        print(f"  - Test Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  - Test Macro F1:   {metrics['macro_f1']:.4f}")
        print(f"  - OOB Accuracy:    {rf.oob_score_:.4f}")
        print(f"  - Train Time:      {fit_time:.2f}s")

        records.append({
            "Configuration": cfg["name"],
            "max_features": str(cfg["max_features"]),
            "max_depth": str(cfg["max_depth"]),
            "criterion": cfg["criterion"],
            "Test Accuracy": metrics["accuracy"],
            "Test Macro F1": metrics["macro_f1"],
            "OOB Accuracy": rf.oob_score_,
            "Train Time (s)": fit_time,
        })

    df_results = pd.DataFrame(records)

    csv_path = os.path.join(fig_dir, "hyperparameter_study.csv")
    tex_path = os.path.join(fig_dir, "hyperparameter_study.tex")

    df_results.to_csv(csv_path, index=False)
    src.dataframe_to_latex_table(
        df_results[["Configuration", "Test Accuracy", "Test Macro F1", "OOB Accuracy", "Train Time (s)"]],
        caption="Hyperparameter Tuning & Subspace Feature Study for Random Forest",
        label="tab:hyperparameter_study",
        filepath=tex_path,
    )

    src.set_plot_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    y_pos = np.arange(len(df_results))
    ax.barh(y_pos, df_results["Test Macro F1"], color="#2b5c8f", edgecolor="#111111", linewidth=0.5, height=0.68)
    ax.set_yticks(y_pos)

    clean_labels = [
        r"Subspace: $\log_2 D$ ($m=5$)",
        r"Subspace: $\sqrt{D}$ ($m=6$, default)",
        r"Subspace: $0.5D$ ($m=21$)",
        r"Subspace: $D$ ($m=42$, bagging)",
        r"Depth: Shallow ($d=8$)",
        r"Depth: Deep ($d=16$)",
        r"Criterion: Entropy ($d=12$)",
    ]
    ax.set_yticklabels(clean_labels if len(clean_labels) == len(df_results) else df_results["Configuration"], fontsize=8.5)
    ax.set_xlabel("Test Macro F1-Score", fontweight="bold", fontsize=9.5)
    ax.set_title("Subspace & Depth Comparison", fontweight="bold", fontsize=10.0)
    ax.set_xlim(0, 0.68)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for idx, val in enumerate(df_results["Test Macro F1"]):
        ax.text(val + 0.008, idx, f"{val:.4f}", va="center", fontsize=8.5, family="serif")

    plt.tight_layout(pad=0.4)
    src.save_figure(fig, os.path.join(fig_dir, "hyperparameter_study.png"))
    src.save_figure(fig, os.path.join(fig_dir, "hyperparameter_study.pdf"))

    print("\n" + "=" * 75)
    print("Hyperparameter Study Results:")
    print("=" * 75)
    print(df_results.to_string(index=False))
    print(f"\nSaved results to {csv_path} and {tex_path}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_hyperparameter_study(quick=args.quick)
