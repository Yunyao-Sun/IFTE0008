"""Plotting and utility functions."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from config import CFG


def plot_power_curve(df: pd.DataFrame) -> None:
    if CFG.TARGET_COL not in df.columns or "wind_speed" not in df.columns:
        print("Power curve plot skipped because target or wind_speed is missing.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df[["wind_speed", CFG.TARGET_COL]].dropna()
    if len(sample) > 20000:
        sample = sample.sample(20000, random_state=CFG.SEED)
    ax.scatter(sample["wind_speed"], sample[CFG.TARGET_COL], s=4, alpha=0.25)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Power curve sample")
    fig.tight_layout()
    path = CFG.RESULTS_DIR / "power_curve_sample.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved power curve plot to {path}")


def plot_ablation_bar(all_results: Dict[str, pd.DataFrame]) -> None:
    if not all_results:
        return

    rows = []
    for name, df in all_results.items():
        rows.append({"experiment": name, "RMSE": df["RMSE"].mean(), "MAE": df["MAE"].mean()})
    summary = pd.DataFrame(rows).sort_values("RMSE")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["experiment"], summary["RMSE"])
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Ablation comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = CFG.RESULTS_DIR / "ablation_rmse_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved ablation bar plot to {path}")
