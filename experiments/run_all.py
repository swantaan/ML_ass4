from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.exp1_baseline_comparison import run_baseline_comparison
from experiments.exp2_ensemble_size import run_ensemble_size_convergence
from experiments.exp3_hyperparameters import run_hyperparameter_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    t_total_start = time.time()
    print("=" * 80)
    print("ASSIGNMENT 4: OPTION 1 (RANDOM FOREST ENSEMBLE) EMPIRICAL SUITE")
    print(f"Mode: {'QUICK (subset verification)' if args.quick else 'FULL (complete dataset)'}")
    print("=" * 80)

    t0 = time.time()
    run_baseline_comparison(quick=args.quick)
    print(f"\n[Completed Experiment 1 in {time.time() - t0:.2f}s]\n")

    t1 = time.time()
    run_ensemble_size_convergence(quick=args.quick)
    print(f"\n[Completed Experiment 2 in {time.time() - t1:.2f}s]\n")

    t2 = time.time()
    run_hyperparameter_study(quick=args.quick)
    print(f"\n[Completed Experiment 3 in {time.time() - t2:.2f}s]\n")

    total_time = time.time() - t_total_start
    print("=" * 80)
    print(f"All experiments completed successfully in {total_time:.2f}s ({total_time / 60:.1f} mins)!")
    print(f"Artifacts saved in: {os.path.join(project_root, 'figures')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
