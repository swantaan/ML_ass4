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
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#dddddd"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 10.5
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9.5
    plt.rcParams["ytick.labelsize"] = 9.5
    plt.rcParams["legend.fontsize"] = 9.5
    plt.rcParams["figure.titlesize"] = 11


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

    short_defaults = [
        "Normal", "Recon", "Backdoor", "DoS", "Exploits",
        "Analysis", "Fuzzers", "Worms", "Shellcode", "Generic"
    ]
    if class_names is None:
        if n_classes == 10:
            class_names = short_defaults
        else:
            class_names = [ATTACK_CATEGORY_MAP.get(i, f"Class {i}") for i in range(n_classes)]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display_data = np.divide(
            cm.astype(np.float32),
            row_sums,
            out=np.zeros_like(cm, dtype=np.float32),
            where=row_sums != 0,
        ) * 100.0
    else:
        display_data = cm.astype(np.float32)

    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    annot_labels = np.empty_like(display_data, dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            val = display_data[i, j]
            if val >= 10.0:
                annot_labels[i, j] = f"{val:.0f}"
            elif val >= 1.0:
                annot_labels[i, j] = f"{val:.1f}"
            elif val >= 0.05:
                annot_labels[i, j] = f"{val:.1f}"
            else:
                annot_labels[i, j] = ""

    sns.heatmap(
        display_data,
        annot=annot_labels,
        fmt="",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        cbar_kws={"shrink": 0.85, "label": "Recall (%)", "pad": 0.02},
        linewidths=0.3,
        linecolor="#cccccc",
        annot_kws={"size": 8.0, "family": "serif"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontweight="bold", fontsize=9.5, labelpad=3)
    ax.set_ylabel("True Label", fontweight="bold", fontsize=9.5, labelpad=3)
    ax.tick_params(axis="both", labelsize=8.5)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout(pad=0.2)

    if filepath:
        save_figure(fig, filepath)

    return fig


def plot_feature_importances(
    feature_names: Sequence[str],
    importances: np.ndarray,
    top_n: int = 15,
    title: str = "Top 15 Predictive Features (MDI)",
    color: str = "#2b5c8f",
    filepath: Optional[str] = None,
) -> plt.Figure:
    set_plot_style()
    order = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in order][::-1]
    top_scores = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(3.5, 3.7))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_scores, color=color, edgecolor="#111111", linewidth=0.5, height=0.68)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=8.5)
    ax.set_xlabel("Relative Importance (MDI)", fontweight="bold", fontsize=9.5)
    ax.set_title(title, fontweight="bold", fontsize=10.0)
    ax.set_xlim(0, max(top_scores) * 1.25)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    for idx, val in enumerate(top_scores):
        ax.text(val + 0.002, idx, f"{val:.4f}", va="center", fontsize=8.0, family="serif")

    plt.tight_layout(pad=0.4)

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
