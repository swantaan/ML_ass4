from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import ATTACK_CATEGORY_MAP


def set_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#dddddd"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.titlesize"] = 12


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300) -> str:
    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    cmap: str = "Blues",
    filepath: Optional[str] = None,
) -> plt.Figure:
    set_plot_style()
    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [ATTACK_CATEGORY_MAP.get(i, f"Class {i}") for i in range(n_classes)]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display_data = np.divide(
            cm.astype(np.float32),
            row_sums,
            out=np.zeros_like(cm, dtype=np.float32),
            where=row_sums != 0,
        )
        fmt = ".2%"
    else:
        display_data = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        display_data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="#cccccc",
        ax=ax,
    )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontweight="bold", labelpad=8)
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if filepath:
        save_figure(fig, filepath)

    return fig


def plot_feature_importances(
    feature_names: Sequence[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importances (Gini MDI)",
    color: str = "#2b5c8f",
    filepath: Optional[str] = None,
) -> plt.Figure:
    set_plot_style()
    order = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in order][::-1]
    top_scores = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(7.5, max(4.5, top_n * 0.28)))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_scores, color=color, edgecolor="#111111", linewidth=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.set_xlabel("Relative Importance (MDI)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for idx, val in enumerate(top_scores):
        ax.text(val + 0.001, idx, f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()

    if filepath:
        save_figure(fig, filepath)

    return fig


def plot_class_distribution(
    y: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    log_scale: bool = True,
    title: str = "Target Class Distribution (attack_cat)",
    color: str = "#346888",
    filepath: Optional[str] = None,
) -> plt.Figure:
    set_plot_style()
    if class_names is None:
        class_names = ATTACK_CATEGORY_MAP

    uniques, counts = np.unique(y, return_counts=True)
    labels = [class_names.get(int(c), f"Class {c}") for c in uniques]
    pcts = (counts / len(y)) * 100

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(labels, counts, color=color, edgecolor="#222222", linewidth=0.7)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Instance Count (Log Scale)", fontweight="bold")
    else:
        ax.set_ylabel("Instance Count", fontweight="bold")

    ax.set_title(title, fontweight="bold", pad=10)
    plt.xticks(rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, cnt, pct in zip(bars, counts, pcts):
        height = bar.get_height()
        ax.annotate(
            f"{cnt:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    if filepath:
        save_figure(fig, filepath)

    return fig


def plot_oob_convergence(
    tree_counts: Sequence[int],
    oob_scores: Sequence[float],
    test_scores: Optional[Sequence[float]] = None,
    title: str = "Random Forest OOB vs Test Accuracy",
    filepath: Optional[str] = None,
) -> plt.Figure:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    ax.plot(
        tree_counts,
        oob_scores,
        marker="o",
        color="#1f77b4",
        label="Out-of-Bag (OOB) Accuracy",
        linewidth=1.8,
    )

    if test_scores is not None:
        ax.plot(
            tree_counts,
            test_scores,
            marker="s",
            color="#ff7f0e",
            linestyle="--",
            label="Test Set Accuracy",
            linewidth=1.8,
        )

    ax.set_xlabel("Number of Estimators (Trees)", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if filepath:
        save_figure(fig, filepath)

    return fig


def dataframe_to_latex_table(
    df: pd.DataFrame,
    caption: str = "Model Performance Summary",
    label: str = "tab:performance_summary",
    column_format: Optional[str] = None,
    filepath: Optional[str] = None,
) -> str:
    n_cols = len(df.columns)
    if column_format is None:
        column_format = "l" + "c" * (n_cols - 1)

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule",
        " & ".join(df.columns) + " \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        formatted_vals = []
        for v in row:
            if isinstance(v, float):
                formatted_vals.append(f"{v:.4f}")
            elif isinstance(v, int):
                formatted_vals.append(f"{v:,}")
            else:
                formatted_vals.append(str(v))
        latex_lines.append(" & ".join(formatted_vals) + " \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    table_code = "\n".join(latex_lines)

    if filepath:
        out_dir = os.path.dirname(filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(table_code)

    return table_code


if __name__ == "__main__":
    from src.data_preprocessing import load_data, preprocess_data

    data_path = os.path.join(project_root, "data", "raw", "networkTraffic.csv")
    raw_df = load_data(data_path).head(5000)
    X, y, fnames, _ = preprocess_data(raw_df, is_train=True)

    os.makedirs(os.path.join(project_root, "figures"), exist_ok=True)

    fig_dist = plot_class_distribution(
        y, filepath=os.path.join(project_root, "figures", "class_distribution_sample.png")
    )
    print("Class distribution plot saved.")

    dummy_importances = np.random.rand(len(fnames))
    dummy_importances /= dummy_importances.sum()
    fig_imp = plot_feature_importances(
        fnames,
        dummy_importances,
        top_n=15,
        filepath=os.path.join(project_root, "figures", "feature_importances_sample.png"),
    )
    print("Feature importances plot saved.")

    dummy_cm = np.random.randint(0, 100, size=(10, 10))
    fig_cm = plot_confusion_matrix(
        dummy_cm,
        filepath=os.path.join(project_root, "figures", "confusion_matrix_sample.png"),
    )
    print("Confusion matrix plot saved.")

    demo_df = pd.DataFrame({
        "Model": ["Decision Tree", "k-NN", "Random Forest"],
        "Accuracy": [0.8083, 0.6450, 0.8172],
        "Macro F1": [0.5399, 0.3193, 0.4823],
        "Train Time (s)": [10.80, 0.0001, 52.53],
    })
    latex_code = dataframe_to_latex_table(
        demo_df, filepath=os.path.join(project_root, "figures", "demo_table.tex")
    )
    print("LaTeX table generated:\n", latex_code)