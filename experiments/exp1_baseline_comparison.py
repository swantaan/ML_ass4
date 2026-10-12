from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src

def run_baseline_comparison(quick: bool = False) -> pd.DataFrame:
    data_path = os.path.join(project_root, "data", "raw", "networkTraffic.csv")
    fig_dir = os.path.join(project_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 75)
    print("Experiment 1: Baseline Comparison (Decision Tree & k-NN vs Random Forest)")
    print("=" * 75)

    raw_df = src.load_data(data_path)
    if quick:
        raw_df = raw_df.head(20000)

    X, y, feature_names, _ = src.preprocess_data(raw_df, is_train=True)
    X_train, X_test, y_train, y_test = src.stratified_train_test_split(
        X, y, test_size=0.20, random_state=69
    )

    print(f"Dataset Size: {X.shape[0]:,} samples x {X.shape[1]} features")
    print(f"Train Set:    {X_train.shape[0]:,} samples")
    print(f"Test Set:     {X_test.shape[0]:,} samples")

    models = {
        "Decision Tree": src.DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=20,
            criterion="gini",
            random_state=69,
        ),
        "k-NN (k=5)": src.KNeighborsClassifier(
            n_neighbors=5,
            metric="euclidean",
            weights="distance",
        ),
        "Random Forest (B=30)": src.RandomForestClassifier(
            n_estimators=30,
            max_features="sqrt",
            max_depth=12,
            min_samples_split=20,
            criterion="gini",
            oob_score=True,
            random_state=69,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        if "k-NN" in name and len(X_train) > 15000:
            sub_rng = np.random.default_rng(69)
            sub_idx = sub_rng.choice(len(X_train), size=15000, replace=False)
            test_sub_idx = sub_rng.choice(len(X_test), size=3000, replace=False)
            X_tr, y_tr = X_train[sub_idx], y_train[sub_idx]
            X_te, y_te = X_test[test_sub_idx], y_test[test_sub_idx]
        else:
            X_tr, y_tr = X_train, y_train
            X_te, y_te = X_test, y_test

        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_time = time.time() - t0

        t1 = time.time()
        y_pred = model.predict(X_te)
        pred_time = time.time() - t1

        metrics = src.compute_metrics(y_te, y_pred, n_classes=10)

        results.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Macro F1": metrics["macro_f1"],
            "Macro Precision": metrics["macro_precision"],
            "Macro Recall": metrics["macro_recall"],
            "Weighted F1": metrics["weighted_f1"],
            "Train Time (s)": fit_time,
            "Inference Time (s)": pred_time,
        })

        clean_slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")
        src.plot_confusion_matrix(
            metrics["confusion_matrix"],
            title=f"Confusion Matrix: {name}",
            normalize=True,
            filepath=os.path.join(fig_dir, f"cm_{clean_slug}.png"),
        )
        src.plot_confusion_matrix(
            metrics["confusion_matrix"],
            title=f"Confusion Matrix: {name}",
            normalize=True,
            filepath=os.path.join(fig_dir, f"cm_{clean_slug}.pdf"),
        )

        if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
            src.plot_feature_importances(
                feature_names,
                model.feature_importances_,
                top_n=15,
                title=f"Feature Importances (MDI): {name}",
                filepath=os.path.join(fig_dir, f"feature_importances_{clean_slug}.png"),
            )
            src.plot_feature_importances(
                feature_names,
                model.feature_importances_,
                top_n=15,
                title=f"Feature Importances (MDI): {name}",
                filepath=os.path.join(fig_dir, f"feature_importances_{clean_slug}.pdf"),
            )

    df_results = pd.DataFrame(results)

    csv_path = os.path.join(fig_dir, "baseline_comparison.csv")
    tex_path = os.path.join(fig_dir, "baseline_comparison.tex")

    df_results.to_csv(csv_path, index=False)
    src.dataframe_to_latex_table(
        df_results,
        caption="Baseline Comparison on UNSW-NB15 Cyberattack Dataset",
        label="tab:baseline_comparison",
        filepath=tex_path,
    )

    print("\n" + "=" * 75)
    print("Baseline Comparison Results:")
    print("=" * 75)
    print(df_results.to_string(index=False))
    print(f"\nSaved results to {csv_path} and {tex_path}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_baseline_comparison(quick=args.quick)
