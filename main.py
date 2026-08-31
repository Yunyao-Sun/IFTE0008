"""
Main entry point for five Penmanshiel ablation experiments.

Usage:
    python main.py
    python main.py --exp 1
    python main.py --exp 3 5
    python main.py --skip 1 2
"""

from __future__ import annotations

import argparse
import traceback

import numpy as np
import torch

try:
    import lightning.pytorch as pl
except Exception:
    import pytorch_lightning as pl

from config import CFG
from src import models
from src.data_loader import load_raw
from src.experiments import run_e1, run_e2, run_e3, run_e4, run_e5
from src.metrics import save_all_summary
from src.preprocessing import preprocess, split_data
from src.utils import plot_ablation_bar, plot_power_curve


def parse_args():
    parser = argparse.ArgumentParser(
        description="Penmanshiel MVMD-TFT ablation experiments")
    parser.add_argument(
        "--exp", nargs="+", type=int, choices=[1, 2, 3, 4, 5],
        help="Run only selected experiments")
    parser.add_argument(
        "--skip", nargs="+", type=int, choices=[1, 2, 3, 4, 5],
        help="Skip selected experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)
    pl.seed_everything(CFG.SEED, workers=True)

    run_all  = set(range(1, 6))
    to_run   = set(args.exp)  if args.exp  else run_all
    to_skip  = set(args.skip) if args.skip else set()
    to_run   = sorted(to_run - to_skip)

    print("=" * 70)
    print("Penmanshiel MVMD-TFT Ablation Experiments")
    print("=" * 70)
    print(f"  Experiments to run : {to_run}")
    print(f"  FREQ               : {CFG.FREQ}  (turbine={CFG.TURBINE_ID})")
    print(f"  WINDOW_SIZE        : {CFG.WINDOW_SIZE} steps = "
          f"{CFG.WINDOW_SIZE * CFG.FREQ_MINUTES / 60:.1f} hours lookback")
    print(f"  HORIZON            : {CFG.HORIZON} steps = "
          f"{CFG.HORIZON * CFG.FREQ_MINUTES / 60:.1f} hours ahead")
    print(f"  STEP               : {CFG.STEP} steps = "
          f"{CFG.STEP * CFG.FREQ_MINUTES} min sliding interval")
    print(f"  MAX_SAMPLES        : {CFG.MAX_SAMPLES_PER_SPLIT}")
    print(f"  BATCH_SIZE         : {CFG.BATCH_SIZE}")
    print(f"  MAX_EPOCHS         : {CFG.MAX_EPOCHS}")
    print("=" * 70)

    df_raw   = load_raw()
    df_clean = preprocess(df_raw)

    if CFG.TARGET_COL not in df_clean.columns:
        print("Target power column not found after preprocessing.")
        print(f"Cleaned data saved to {CFG.RESULTS_DIR}")
        return

    plot_power_curve(df_clean)
    train_df, val_df, test_df = split_data(df_clean)

    from src.dataset_builder import TARGET_SCALER, reset_global_state
    reset_global_state()
    TARGET_SCALER.fit(train_df[CFG.TARGET_COL].values)
    models.set_global_normalizer(TARGET_SCALER)

    experiment_functions = {
        1: run_e1,
        2: run_e2,
        3: run_e3,
        4: run_e4,
        5: run_e5,
    }

    all_results = {}
    e4_selector = None

    for exp_id in to_run:
        label = CFG.EXP_NAMES[exp_id]
        print("\n" + "=" * 70)
        print(f"Running {label}")
        print("=" * 70)

        try:
            if exp_id == 5:
                result = experiment_functions[exp_id](
                    train_df, val_df, test_df, e4_selector)
            else:
                result = experiment_functions[exp_id](
                    train_df, val_df, test_df)

            all_results[label] = result["metrics"]

            if exp_id == 4:
                e4_selector = result.get("selector")

        except Exception as exc:
            print(f"\nExperiment {label} failed: {exc}")
            traceback.print_exc()
            print("Continuing to next experiment...\n")

    if all_results:
        print("\nSaving final summary...")
        save_all_summary(all_results)
        plot_ablation_bar(all_results)

    print(f"\nDone. Results directory: {CFG.RESULTS_DIR}")


if __name__ == "__main__":
    main()
