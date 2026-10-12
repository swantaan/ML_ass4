from __future__ import annotations

from src.data_preprocessing import (
    ATTACK_CATEGORY_MAP,
    ATTACK_NAME_TO_ID,
    ID_COLUMN,
    NOMINAL_COLUMNS,
    TARGET_COLUMN,
    NetworkTrafficPreprocessor,
    load_data,
    preprocess_data,
)
from src.baselines import (
    DecisionTreeClassifier,
    DecisionTreeNode,
    KNeighborsClassifier,
    evaluate_baseline,
)
from src.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    get_rf_configurations,
)
from src.evaluation import (
    StratifiedKFold,
    accuracy_score,
    classification_report,
    compute_metrics,
    confusion_matrix,
    cross_validate,
    cv_summary_dataframe,
    evaluate_oob,
    macro_precision_recall_f1,
    stratified_train_test_split,
)
from src.utils import (
    dataframe_to_latex_table,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_oob_convergence,
    save_figure,
    set_plot_style,
)

__all__ = [
    "ATTACK_CATEGORY_MAP",
    "ATTACK_NAME_TO_ID",
    "NOMINAL_COLUMNS",
    "ID_COLUMN",
    "TARGET_COLUMN",
    "NetworkTrafficPreprocessor",
    "load_data",
    "preprocess_data",
    "stratified_train_test_split",
    "DecisionTreeNode",
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
    "evaluate_baseline",
    "BaggingClassifier",
    "RandomForestClassifier",
    "get_rf_configurations",
    "StratifiedKFold",
    "accuracy_score",
    "confusion_matrix",
    "compute_metrics",
    "classification_report",
    "evaluate_oob",
    "cross_validate",
    "cv_summary_dataframe",
    "macro_precision_recall_f1",
    "set_plot_style",
    "save_figure",
    "plot_confusion_matrix",
    "plot_feature_importances",
    "plot_class_distribution",
    "plot_oob_convergence",
    "dataframe_to_latex_table",
]
