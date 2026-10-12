from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src


def run_ensemble_size_convergence(quick: bool = False) -> pd.DataFrame:
    data_path = os.path.join(project_root, "data", "raw", "networkTraffic.csv")
    fig_dir = os.path.join(project_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 75)
    print("Experiment 2: Ensemble Size Convergence Study (Number of Trees)")
    print("=" * 75)

    raw_df = src.load_data(data_path)
    if quick:
        raw_df = raw_df.head(20000)
        tree_counts = [5, 10, 15, 20]
    else:
        tree_counts = [5, 10, 20, 30, 50, 75]

    X, y, feature_names, _ = src.preprocess_data(raw_df, is_train=True)
    X_train, X_test, y_train, y_test = src.stratified_train_test_split(
        X, y, test_size=0.20, random_state=69
    )

    records = []

    for b in tree_counts:
        print(f"\nTraining Random Forest with B = {b} estimators...")
        rf = src.RandomForestClassifier(
            n_estimators=b,
            max_features="sqrt",
            max_depth=12,
            min_samples_split=20,
            criterion="gini",
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

        oob_metrics = (
            src.evaluate_oob(rf.oob_decision_function_, y_train)
            if rf.oob_decision_function_ is not None
            else {"oob_accuracy": 0.0, "oob_macro_f1": 0.0}
        )

        print(f"  - OOB Accuracy:      {oob_metrics['oob_accuracy']:.4f}")
        print(f"  - Test Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  - Test Macro F1:     {metrics['macro_f1']:.4f}")
        print(f"  - Train Time:        {fit_time:.2f}s")

        records.append({
            "Trees (B)": b,
            "OOB Accuracy": oob_metrics["oob_accuracy"],
            "OOB Macro F1": oob_metrics["oob_macro_f1"],
            "Test Accuracy": metrics["accuracy"],
            "Test Macro F1": metrics["macro_f1"],
            "Train Time (s)": fit_time,
            "Inference Time (s)": pred_time,
        })

    df_results = pd.DataFrame(records)

    csv_path = os.path.join(fig_dir, "ensemble_size_convergence.csv")
    tex_path = os.path.join(fig_dir, "ensemble_size_convergence.tex")

    df_results.to_csv(csv_path, index=False)
    src.dataframe_to_latex_table(
        df_results,
        caption="Effect of Ensemble Size (B) on Random Forest Performance",
        label="tab:ensemble_size_convergence",
        filepath=tex_path,
    )

    src.set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(
        df_results["Trees (B)"],
        df_results["OOB Accuracy"],
        marker="o",
        label="OOB Accuracy",
        color="#1f77b4",
        linewidth=1.8,
    )
    ax1.plot(
        df_results["Trees (B)"],
        df_results["Test Accuracy"],
        marker="s",
        label="Test Accuracy",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8,
    )
    ax1.set_xlabel("Number of Estimators (B)", fontweight="bold")
    ax1.set_ylabel("Accuracy", fontweight="bold")
    ax1.set_title("OOB vs Test Accuracy Convergence", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(
        df_results["Trees (B)"],
        df_results["Train Time (s)"],
        marker="^",
        label="Training Wall Time",
        color="#2ca02c",
        linewidth=1.8,
    )
    ax2.set_xlabel("Number of Estimators (B)", fontweight="bold")
    ax2.set_ylabel("Training Time (seconds)", fontweight="bold")
    ax2.set_title("Computational Cost Scaling", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    src.save_figure(fig, os.path.join(fig_dir, "ensemble_size_convergence.png"))
    src.save_figure(fig, os.path.join(fig_dir, "ensemble_size_convergence.pdf"))

    print("\n" + "=" * 75)
    print("Ensemble Size Study Summary:")
    print("=" * 75)
    print(df_results.to_string(index=False))
    print(f"\nSaved results to {csv_path} and {tex_path}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_ensemble_size_convergence(quick=args.quick)
