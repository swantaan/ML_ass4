# Empirical Evaluation of Random Forest Ensembles for Multi-Class Network Intrusion Detection

This repository implements and evaluates Breiman's **Random Forest** algorithm and baseline classifiers (**Classification and Regression Trees** and **$k$-Nearest Neighbours**) on the UNSW-NB15 network intrusion dataset for multi-class cyberattack classification.

All models, data transformations, out-of-bag validation metrics, and visualization utilities are implemented **from scratch in pure NumPy and Python** without the use of third-party machine learning libraries (such as scikit-learn).

---

## Key Implementations

- **Random Forest Classifier (`src/ensemble.py`)**:
  - Independent bootstrap aggregation (bagging) with replacement.
  - Random subspace feature selection ($m = \lfloor \sqrt{D} \rfloor$, $m = \lfloor \log_2 D \rfloor$, or custom feature fraction).
  - Out-of-bag (OOB) error estimation and probability tracking across bootstrap omissions.
  - Multi-threaded ensemble training via Python's `ThreadPoolExecutor`.
  - Mean Decrease in Impurity (MDI) feature importance calculation.

- **Decision Tree Classifier (`src/baselines.py`)**:
  - Recursive binary tree partitioning supporting both Gini Impurity and Shannon Entropy split criteria.
  - Quantile-based candidate split discretization (`n_bins=32`) for accelerated split searching across continuous features.
  - Configurable pre-pruning constraints (`max_depth`, `min_samples_split`, `min_samples_leaf`).

- **$k$-Nearest Neighbours Classifier (`src/baselines.py`)**:
  - Distance-weighted and uniform majority voting.
  - Chunked/batched pairwise metric evaluation to avoid out-of-memory bottlenecks.

- **Preprocessing Pipeline (`src/data_preprocessing.py`)**:
  - Explicit categorical encoding for network protocols, services, and connection states.
  - Frequency thresholding for low-frequency protocol categories (`proto_threshold=0.001`).
  - Handling missing service records (`'?'` mapped to `'unknown'`).
  - Removal of non-predictive observation identifiers (`id`) and leakage channels.

- **Evaluation & Visualisation (`src/evaluation.py`, `src/utils.py`)**:
  - Multi-class confusion matrices, per-class metrics, macro/weighted averages.
  - Publication-ready vector plots and automated LaTeX table generation.

---

## Repository Structure

```text
.
├── 26895064RW741assignment4.pdf   # Submitted formal report (PDF)
├── assignment4.pdf                # Assignment specification
├── data/
│   ├── processed/                 # Intermediate processed caches (if generated)
│   └── raw/
│       ├── attack_category_map.csv # Attack category identifier mapping
│       ├── features.csv            # UNSW-NB15 feature descriptions
│       └── networkTraffic.csv      # Complete dataset (257,673 records, 45 attributes)
├── experiments/
│   ├── exp1_baseline_comparison.py # Experiment 1: Baseline comparison (DT, k-NN, RF)
│   ├── exp2_ensemble_size.py       # Experiment 2: Convergence study over ensemble size B
│   ├── exp3_hyperparameters.py     # Experiment 3: Feature subspace & depth tuning
│   └── run_all.py                  # Master experimental runner
├── figures/                       # Generated figures, confusion matrices, and tables
├── report/
│   ├── ass4.tex                   # IEEE conference template LaTeX source
│   ├── ass4.pdf                   # Compiled PDF report
│   └── writingRules.pdf           # Writing guidelines
├── requirements.txt               # Minimal Python dependencies
├── src/
│   ├── __init__.py                # Package exports
│   ├── baselines.py               # Custom Decision Tree and k-NN implementations
│   ├── data_preprocessing.py      # Cleaning and transformation pipeline
│   ├── ensemble.py                # Bagging and Random Forest implementations
│   ├── evaluation.py              # Metrics, stratified splitting, and OOB tracking
│   └── utils.py                   # Plotting utilities and LaTeX formatting
└── README.md                      # Repository documentation
```

---

## Installation

### Prerequisites
- Linux OS (recommended: Ubuntu 22.04 / 24.04 LTS)
- Python $\ge$ 3.10

### Setup
Clone the repository and install the minimal dependencies:

```bash
git clone https://github.com/swantaan/ML_ass4.git
cd ML_ass4
pip install -r requirements.txt
```

---

## Running Experiments

All experiments can be executed either in full (complete dataset) or quick verification mode (subset):

### 1. Run Complete Suite
To run all three experiments sequentially:

```bash
python3 experiments/run_all.py
```

To run a fast verification pass across all experiments:

```bash
python3 experiments/run_all.py --quick
```

### 2. Run Individual Experiments

- **Experiment 1 (Baseline Comparison)**:
  ```bash
  python3 experiments/exp1_baseline_comparison.py
  ```
  Evaluates Decision Tree, $k$-NN, and Random Forest ($B=30$) baselines, generating confusion matrices and LaTeX comparison tables.

- **Experiment 2 (Ensemble Size Convergence)**:
  ```bash
  python3 experiments/exp2_ensemble_size.py
  ```
  Evaluates ensemble convergence across $B \in \{5, 10, 20, 30, 50, 75\}$, comparing test accuracy against out-of-bag estimates and tracking wall-clock training time.

- **Experiment 3 (Hyperparameter & Subspace Study)**:
  ```bash
  python3 experiments/exp3_hyperparameters.py
  ```
  Analyzes feature subspace constraints ($m = \lfloor \log_2 D \rfloor$, $m = \lfloor \sqrt{D} \rfloor$, $m = 0.5D$, $m = D$), tree depth limits ($d \in \{8, 12, 16\}$), and split criteria (Gini vs Entropy).

All generated figures (`.pdf`, `.png`), `.csv` metrics, and `.tex` tables are saved automatically to `figures/`.

---

## Empirical Results Summary

Evaluated on 257,673 records from the UNSW-NB15 dataset partitioned into an 80% stratified training set and a 20% held-out test set (seed 69):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k$-Nearest Neighbours** ($k=5$) | 65.13% | 34.61% | 34.02% | 31.96% | 61.64% | 0.001 s |
| **Decision Tree** ($d=12$) | 80.97% | 72.63% | 46.52% | 54.34% | 78.69% | 12.35 s |
| **Random Forest** ($B=30$, $m=\lfloor\sqrt{D}\rfloor$) | **81.83%** | **84.52%** | **44.97%** | **50.62%** | **78.66%** | 56.40 s |

### Key Takeaways
1. **Variance Reduction**: The Random Forest ensemble outperforms both individual decision trees and $k$-NN baselines in overall test accuracy (81.83%) and macro precision (84.52%).
2. **OOB Convergence**: Out-of-bag accuracy converged to within 0.20% of test generalization accuracy as $B \ge 30$, demonstrating OOB estimation as an effective substitute for $K$-fold cross-validation.
3. **Subspace Efficiency**: Random feature subspace selection ($m = \lfloor \sqrt{D} \rfloor$) reduced training execution time by 77.5% compared to full bagging ($m = D$) while maintaining competitive classification accuracy.
4. **Key Features**: Destination service connection counts (`ct_srv_dst`), source time-to-live (`sttl`), and destination download throughput (`dload`) exhibited the highest discriminative importance via Mean Decrease in Impurity.

---

## Report

The formal technical report is written in IEEE conference format (2-column, 10pt) and is located at:
- Source: [`report/ass4.tex`](file:///home/christiaan/Desktop/ML/assignment4/report/ass4.tex)
- PDF: [`26895064RW741assignment4.pdf`](file:///home/christiaan/Desktop/ML/assignment4/26895064RW741assignment4.pdf) (and [`report/ass4.pdf`](file:///home/christiaan/Desktop/ML/assignment4/report/ass4.pdf))
