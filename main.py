"""
Main entry point for the pointwise-data-driven economic value / VaR analysis.

One command does the ENTIRE pipeline, no manual multi-step process needed:
    python main.py

Full sequence run automatically inside this one command:
    1. Check/auto-fetch System Price (fetch_prices.py)
    2. Check/auto-fetch MID (fetch_mid.py)
    3. Check/auto-build combined real price table (build_combined_price_table.py)
    4. Scan inputs/pointwise/, compute VaR/CVaR + paired incremental loss
       for every {turbine}_{freq}_{experiment}_pointwise.csv found
    5. If inputs/ablation/ contains per-combo RMSE summary CSVs, auto-combine
       them and join with the economic results (RMSE improvement vs
       economic savings, in one table)
    6. Auto-generate all figures (heatmap, loss distributions, incremental
       loss distribution, loss-vs-spread scatter)

Inputs you need to provide (everything else is auto-generated/auto-fetched):
    inputs/pointwise/*.csv    -- required. {turbine}_{freq}_{experiment}_pointwise.csv
    inputs/ablation/*.csv     -- optional. Per-combo RMSE summary CSVs (5 rows:
                                  E1-E5). Filename just needs a "WTn" and an
                                  "Nmin" pattern somewhere in it, e.g.
                                  "WT1 10 min all_5_ablation_summary.csv" works.

Outputs (all in ./outputs/):
    economic_value_summary_v2.csv     -- per (turbine, granularity, experiment)
                                          VaR95/99, CVaR95/99, expected cost
    savings_vs_E1_baseline.csv        -- level-VaR/CVaR difference vs E1
                                          (VaR(E_m) - VaR(E1), see note below)
    incremental_loss_vs_E1.csv        -- PAIRED per-scenario delta cost
                                          (L_m - L_E1 for matching group_id),
                                          with VaR95/99(delta), CVaR95/99(delta)
                                          -- this is the statistically correct
                                          way to ask "how much does switching
                                          from E1 to E_m change my risk?"
    raw_costs/{turbine}_{freq}_{experiment}_costs.csv  -- group_id + scenario_loss
    accuracy_vs_economic_value.csv    -- only produced if inputs/ablation/ has
                                          data: RMSE improvement vs E1 joined
                                          with economic savings vs E1, one row
                                          per (turbine, granularity, experiment)
    figures/                          -- heatmap + loss distribution charts
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data_loader import discover_pointwise_files, load_pointwise, freq_to_minutes
from imbalance_from_pointwise import compute_sample_costs, find_valid_start_indices, SCENARIO_HORIZON_PERIODS
from var_risk import summarize_costs

SAVE_RAW_COSTS = True
RAW_COSTS_DIR = config.OUTPUT_DIR / "raw_costs"

EXP_ORDER = ["E1_raw_tft", "E2_global_mvmd_tft", "E3_rolling_mvmd_tft",
             "E4_rolling_mvmd_en_lstm", "E5_rolling_mvmd_en_tft"]
EXP_LABELS = {
    "E1_raw_tft": "E1 Raw+TFT",
    "E2_global_mvmd_tft": "E2 Global MVMD+TFT",
    "E3_rolling_mvmd_tft": "E3 Rolling MVMD+TFT",
    "E4_rolling_mvmd_en_lstm": "E4 Rolling MVMD+EN+LSTM",
    "E5_rolling_mvmd_en_tft": "E5 Rolling MVMD+EN+TFT",
}
EXP_COLORS = {
    "E1_raw_tft": "#888888", "E2_global_mvmd_tft": "#2ca02c",
    "E3_rolling_mvmd_tft": "#d62728", "E4_rolling_mvmd_en_lstm": "#9467bd",
    "E5_rolling_mvmd_en_tft": "#1f77b4",
}
FREQ_ORDER = ["10min", "20min", "30min"]


# --------------------------------------------------------------------------
# Auto-download / auto-build real price data (only needed for PRICE_MODE="real_mid")
# --------------------------------------------------------------------------
def ensure_real_price_data() -> pd.DataFrame:
    if not config.PRICE_CSV.exists():
        print(f"System Price file not found ({config.PRICE_CSV}), fetching automatically...")
        from fetch_prices import fetch_and_save_prices
        config.PRICE_CSV.parent.mkdir(parents=True, exist_ok=True)
        fetch_and_save_prices(output_file=str(config.PRICE_CSV))
    else:
        print(f"Found existing System Price file: {config.PRICE_CSV}")

    mid_csv_local = Path("./inputs/elexon_mid_2020_2022.csv")
    if not mid_csv_local.exists():
        print(f"MID file not found ({mid_csv_local}), fetching automatically...")
        mid_csv_local.parent.mkdir(parents=True, exist_ok=True)
        from fetch_mid import fetch_and_save_mid
        fetch_and_save_mid(output_file=str(mid_csv_local))
    else:
        print(f"Found existing MID file: {mid_csv_local}")

    if not config.REAL_PRICE_CSV.exists():
        print(f"Combined price table not found ({config.REAL_PRICE_CSV}), building automatically...")
        from build_combined_price_table import build_combined_price_table
        build_combined_price_table(
            system_price_csv=config.PRICE_CSV, mid_csv=mid_csv_local,
            output_csv=config.REAL_PRICE_CSV)
    else:
        print(f"Found existing combined price table: {config.REAL_PRICE_CSV}")

    price_df = pd.read_csv(config.REAL_PRICE_CSV)
    price_df["startTime"] = pd.to_datetime(price_df["startTime"], utc=True, errors="coerce")
    price_df["settlementDate"] = pd.to_datetime(price_df["settlementDate"])
    price_df = price_df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)

    spread = price_df["marketIndexPrice"] - price_df["systemPrice"]
    print(f"  MID - System spread: mean={spread.mean():.2f}  std={spread.std():.2f}  "
          f"min={spread.min():.2f}  max={spread.max():.2f}")
    if spread.abs().mean() < 0.5:
        raise RuntimeError(
            "marketIndexPrice and systemPrice are essentially identical -- check that "
            "the two source CSVs are genuinely different price series, not duplicates."
        )
    return price_df


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_var_cvar_heatmap(summary_df: pd.DataFrame, out_dir: Path):
    if summary_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, title in zip(axes, ["VaR_95", "CVaR_95"], ["VaR\u2089\u2085 (\u00a3)", "CVaR\u2089\u2085 (\u00a3)"]):
        pivot = summary_df.pivot_table(index="experiment", columns="granularity", values=metric, aggfunc="mean")
        pivot = pivot.reindex(index=[e for e in EXP_ORDER if e in pivot.index],
                               columns=[f for f in FREQ_ORDER if f in pivot.columns])
        if pivot.empty:
            continue
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=11)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([EXP_LABELS.get(e, e) for e in pivot.index], fontsize=10)
        finite_vals = pivot.values[~np.isnan(pivot.values)]
        mean_val = finite_vals.mean() if len(finite_vals) else 0
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = "white" if val > mean_val else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
                else:
                    ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=9)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Granularity", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Experiment", fontsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("\u00a3 (lower = better)", fontsize=9)
    fig.suptitle("Average VaR\u2089\u2085 / CVaR\u2089\u2085 by Experiment and Granularity", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "var_cvar_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_loss_distribution(raw_costs_dir: Path, freq: str, out_dir: Path):
    available_exps, exp_costs = [], {}
    for exp in EXP_ORDER:
        files = list(raw_costs_dir.glob(f"*_{freq}_{exp}_costs.csv"))
        if not files:
            continue
        pooled = pd.concat([pd.read_csv(f)["scenario_loss"] for f in files], ignore_index=True)
        if len(pooled) == 0:
            continue
        exp_costs[exp] = pooled
        available_exps.append(exp)
    if not available_exps:
        return

    fig, axes = plt.subplots(len(available_exps), 1, figsize=(9, 2.4 * len(available_exps)), sharex=True)
    if len(available_exps) == 1:
        axes = [axes]
    for ax, exp in zip(axes, available_exps):
        costs = exp_costs[exp]
        var95 = np.quantile(costs, 0.95)
        cvar95 = costs[costs >= var95].mean()
        mean_cost = costs.mean()
        ax.hist(costs, bins=40, color=EXP_COLORS.get(exp, "#1f77b4"), alpha=0.65, edgecolor="white", linewidth=0.3)
        ax.axvline(mean_cost, color="black", linewidth=1.2, label=f"Mean = \u00a3{mean_cost:.1f}")
        ax.axvline(var95, color="darkorange", linestyle="--", linewidth=1.5, label=f"VaR\u2089\u2085 = \u00a3{var95:.1f}")
        ax.axvline(cvar95, color="darkred", linestyle=":", linewidth=1.8, label=f"CVaR\u2089\u2085 = \u00a3{cvar95:.1f}")
        ax.axvspan(var95, costs.max(), color="darkred", alpha=0.08)
        ax.set_ylabel(EXP_LABELS.get(exp, exp), fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.tick_params(labelsize=9)
    axes[-1].set_xlabel("Scenario cost (\u00a3, pooled across turbines)", fontsize=11)
    fig.suptitle(f"Loss distribution per 8-hour forecast scenario ({freq})", fontsize=13, y=0.995)
    plt.tight_layout()
    out_path = out_dir / f"loss_distribution_{freq}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_incremental_loss_distribution(incr_df: pd.DataFrame, out_dir: Path):
    if incr_df.empty:
        return
    exps = [e for e in EXP_ORDER if e != "E1_raw_tft" and e in incr_df["experiment"].unique()]
    if not exps:
        return
    fig, axes = plt.subplots(len(exps), 1, figsize=(9, 2.4 * len(exps)), sharex=True)
    if len(exps) == 1:
        axes = [axes]
    for ax, exp in zip(axes, exps):
        deltas = incr_df.loc[incr_df["experiment"] == exp, "delta_cost"]
        var95 = np.quantile(deltas, 0.95)
        mean_delta = deltas.mean()
        ax.hist(deltas, bins=40, color=EXP_COLORS.get(exp, "#1f77b4"), alpha=0.65, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=1, linestyle="-")
        ax.axvline(mean_delta, color="green" if mean_delta < 0 else "red", linewidth=1.5,
                    label=f"Mean \u0394 = \u00a3{mean_delta:.1f}")
        ax.axvline(var95, color="darkorange", linestyle="--", linewidth=1.5, label=f"VaR\u2089\u2085(\u0394) = \u00a3{var95:.1f}")
        ax.set_ylabel(EXP_LABELS.get(exp, exp), fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(labelsize=9)
    axes[-1].set_xlabel("\u0394 cost vs E1 (\u00a3, negative = cheaper than E1)", fontsize=11)
    fig.suptitle("Paired incremental loss distribution vs E1 baseline\n"
                  "(\u0394L = L_experiment \u2212 L_E1, same scenario, same price path)", fontsize=13, y=1.0)
    plt.tight_layout()
    out_path = out_dir / "incremental_loss_vs_E1.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def plot_loss_vs_spread_scatter(raw_costs_dir: Path, out_dir: Path):
    """
    Descriptive scatter of scenario loss vs the mean price spread
    (MID - SystemPrice) sampled for that scenario's price path. Only
    meaningful in PRICE_MODE="real_mid" (mean_spread is NaN otherwise).
    This does NOT claim any real-world temporal correlation between the
    forecast error and the spread -- price paths are randomly sampled from
    history, not date-matched. It illustrates how much the SAME forecast
    error can swing in economic terms depending on which historical price
    regime it happens to be revalued against.
    """
    files = list(raw_costs_dir.glob("*_costs.csv"))
    if not files:
        return
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "mean_spread" not in df.columns or df["mean_spread"].isna().all():
            continue
        parts = f.stem.replace("_costs", "").split("_", 2)
        if len(parts) < 3:
            continue
        df["experiment"] = parts[2]
        frames.append(df)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for exp in EXP_ORDER:
        subset = combined[combined["experiment"] == exp]
        if subset.empty:
            continue
        ax.scatter(subset["mean_spread"], subset["scenario_loss"],
                    s=12, alpha=0.5, color=EXP_COLORS.get(exp, "#1f77b4"),
                    label=EXP_LABELS.get(exp, exp))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean sampled price spread over scenario (MID \u2212 SystemPrice, \u00a3/MWh)", fontsize=11)
    ax.set_ylabel("Scenario loss (\u00a3)", fontsize=11)
    ax.set_title("Scenario loss vs sampled price spread\n"
                  "(descriptive only -- price paths are randomly sampled, not date-matched to the forecast)",
                  fontsize=12)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path = out_dir / "loss_vs_spread_scatter.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


def auto_merge_accuracy_and_economics():
    """
    If inputs/ablation/ contains per-turbine-per-granularity RMSE summary
    CSVs (5 rows each: E1-E5), automatically combine them and merge with
    the PAIRED incremental-loss results (incremental_loss_vs_E1.csv), not
    the level-difference savings_vs_E1_baseline.csv -- the paired delta is
    the statistically correct object to relate RMSE improvement to risk
    change, since VaR(E1)-VaR(E_m) != VaR(E1-E_m) in general. The mean
    saving figure is unaffected by this choice (E[L_E1]-E[L_m] ==
    E[L_E1-L_m] either way), but VaR_95/CVaR_95 of the delta only exist in
    the paired table.
    Produces outputs/accuracy_vs_economic_value.csv and two figures.
    Silently does nothing if inputs/ablation/ doesn't exist or is empty.
    """
    import glob
    import re

    ablation_dir = config.POINTWISE_DIR.parent / "ablation"
    if not ablation_dir.exists():
        return
    files = sorted(glob.glob(str(ablation_dir / "*.csv")))
    if not files:
        return

    print(f"\nFound {len(files)} files in {ablation_dir}, auto-combining RMSE summary...")

    turbine_pattern = re.compile(r"(WT\d+)", re.IGNORECASE)
    freq_pattern = re.compile(r"(\d+)\s*min", re.IGNORECASE)

    frames = []
    for path in files:
        name = Path(path).name
        t_match, f_match = turbine_pattern.search(name), freq_pattern.search(name)
        if not t_match or not f_match:
            print(f"  [skip] couldn't parse turbine/granularity from: {name}")
            continue
        df = pd.read_csv(path)
        df.insert(0, "turbine", t_match.group(1).upper())
        df.insert(1, "granularity", f"{f_match.group(1)}min")
        frames.append(df)

    if not frames:
        print("  No files matched the expected naming pattern, skipping accuracy-economics merge.")
        return

    ablation_df = pd.concat(frames, ignore_index=True)
    print(f"  Combined {len(frames)} files into {len(ablation_df)} rows")

    rmse_col = next((c for c in ["RMSE_mean", "RMSE", "RMSE(kW)"] if c in ablation_df.columns), None)
    if rmse_col is None:
        print(f"  No RMSE column found among RMSE_mean/RMSE/RMSE(kW), skipping accuracy-economics merge.")
        return
    ablation_df = ablation_df.rename(columns={rmse_col: "RMSE"})

    incr_path = config.OUTPUT_DIR / "incremental_loss_vs_E1.csv"
    if not incr_path.exists():
        print(f"  {incr_path} not found, skipping accuracy-economics merge.")
        return
    incr_df = pd.read_csv(incr_path)

    merged = incr_df.merge(
        ablation_df[["turbine", "granularity", "experiment", "RMSE"]],
        on=["turbine", "granularity", "experiment"], how="left")

    e1_rmse = ablation_df[ablation_df["experiment"] == "E1_raw_tft"][
        ["turbine", "granularity", "RMSE"]].rename(columns={"RMSE": "RMSE_E1"})
    merged = merged.merge(e1_rmse, on=["turbine", "granularity"], how="left")
    merged["RMSE_reduction_pct_vs_E1"] = (
        (merged["RMSE_E1"] - merged["RMSE"]) / merged["RMSE_E1"] * 100
    )
    # mean_delta_cost is L_m - L_E1 (positive = worse); flip sign so positive
    # means "saving vs E1", which is the intuitive direction for the scatter plot
    merged["mean_economic_saving_vs_E1"] = -merged["mean_delta_cost"]

    cols = ["turbine", "granularity", "experiment", "RMSE_reduction_pct_vs_E1",
            "mean_economic_saving_vs_E1", "VaR_95_delta", "CVaR_95_delta",
            "VaR_99_delta", "CVaR_99_delta"]
    merged = merged[cols]

    out_path = config.OUTPUT_DIR / "accuracy_vs_economic_value.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {out_path}  (RMSE improvement joined with PAIRED economic delta -- use this for Results)")

    plot_rmse_vs_economic_value(merged, config.OUTPUT_DIR / "figures")


def plot_rmse_vs_economic_value(merged: pd.DataFrame, out_dir: Path):
    """
    Two scatter plots connecting forecast accuracy improvement to economic
    outcome, one row per (turbine, granularity, experiment) with E1 excluded
    (RMSE_reduction and delta are always 0 for E1 vs itself).
    """
    if merged.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_labels = {
        "E2_global_mvmd_tft": "E2 Global MVMD+TFT", "E3_rolling_mvmd_tft": "E3 Rolling MVMD+TFT",
        "E4_rolling_mvmd_en_lstm": "E4 Rolling MVMD+EN+LSTM", "E5_rolling_mvmd_en_tft": "E5 Rolling MVMD+EN+TFT",
    }
    exp_colors = {
        "E2_global_mvmd_tft": "#2ca02c", "E3_rolling_mvmd_tft": "#d62728",
        "E4_rolling_mvmd_en_lstm": "#9467bd", "E5_rolling_mvmd_en_tft": "#1f77b4",
    }

    # --- Plot 1: RMSE reduction vs mean economic saving ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for exp, label in exp_labels.items():
        subset = merged[merged["experiment"] == exp]
        if subset.empty:
            continue
        ax.scatter(subset["RMSE_reduction_pct_vs_E1"], subset["mean_economic_saving_vs_E1"],
                    s=40, alpha=0.75, color=exp_colors.get(exp), label=label, edgecolors="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("RMSE reduction vs E1 (%)", fontsize=11)
    ax.set_ylabel("Mean economic saving vs E1 (\u00a3/8h scenario)", fontsize=11)
    ax.set_title("Does forecast accuracy improvement translate into economic value?\n"
                  "(top-right = more accurate AND cheaper; bottom-right = more accurate but costlier)",
                  fontsize=11)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    p1 = out_dir / "rmse_vs_mean_saving.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {p1}")

    # --- Plot 2: RMSE reduction vs paired CVaR95(delta) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for exp, label in exp_labels.items():
        subset = merged[merged["experiment"] == exp]
        if subset.empty:
            continue
        ax.scatter(subset["RMSE_reduction_pct_vs_E1"], subset["CVaR_95_delta"],
                    s=40, alpha=0.75, color=exp_colors.get(exp), label=label, edgecolors="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("RMSE reduction vs E1 (%)", fontsize=11)
    ax.set_ylabel("CVaR\u2089\u2085 of paired \u0394loss vs E1 (\u00a3, lower = safer tail)", fontsize=11)
    ax.set_title("Does more accurate forecasting also reduce tail risk?\n"
                  "(CVaR\u2089\u2085(\u0394L): average loss in the worst 5% of scenario-level comparisons vs E1)",
                  fontsize=11)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    p2 = out_dir / "rmse_vs_cvar95_delta.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {p2}")


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    price_mode_labels = {
        "fixed": f"fixed symmetric penalty (\u00a3{config.FIXED_ERROR_PENALTY}/MWh)",
        "real_mid": "real Elexon Market Index Data vs System Price",
    }
    print(f"Price mode: {price_mode_labels.get(config.PRICE_MODE, config.PRICE_MODE)}")

    price_df, valid_starts = None, None
    if config.PRICE_MODE == "real_mid":
        price_df = ensure_real_price_data()
        valid_starts = find_valid_start_indices(price_df, SCENARIO_HORIZON_PERIODS)
        print(f"  {len(valid_starts)} valid contiguous 8-hour starting positions "
              f"out of {len(price_df)} total settlement periods")

    print(f"\nScanning: {config.POINTWISE_DIR}")
    files = discover_pointwise_files()
    if not files:
        print(f"No pointwise files found in {config.POINTWISE_DIR}. "
              f"Make sure filenames follow: {{turbine}}_{{freq}}_{{experiment}}_pointwise.csv")
        return
    print(f"Found {len(files)} pointwise files\n")

    # group files by (turbine, granularity) so E1..E5 can be paired for incremental loss
    combos = {}
    for f in files:
        combos.setdefault((f.turbine, f.freq), {})[f.experiment] = f

    summary_rows = []
    incremental_rows = []

    for (turbine, freq), exp_files in sorted(combos.items()):
        print(f"=== {turbine} / {freq} ===")
        step_minutes = freq_to_minutes(freq)
        exp_cost_dfs = {}

        for exp_name in EXP_ORDER:
            if exp_name not in exp_files:
                continue
            f = exp_files[exp_name]
            try:
                df = load_pointwise(f.path)
            except Exception as exc:
                print(f"  [{exp_name}] skipped (failed to load): {exc}")
                continue
            try:
                cost_df = compute_sample_costs(
                    df, step_minutes, turbine, freq,
                    price_df=price_df, valid_start_indices=valid_starts)
            except ValueError as exc:
                print(f"  [{exp_name}] skipped (data integrity check failed): {exc}")
                continue

            if len(cost_df) == 0:
                continue
            exp_cost_dfs[exp_name] = cost_df

            if SAVE_RAW_COSTS:
                RAW_COSTS_DIR.mkdir(parents=True, exist_ok=True)
                cost_df.to_csv(RAW_COSTS_DIR / f"{turbine}_{freq}_{exp_name}_costs.csv", index=False)

            risk_metrics = summarize_costs(cost_df["scenario_loss"].to_numpy())
            print(f"  [{exp_name}] n_samples={risk_metrics['n_samples']}  "
                  f"expected_loss={risk_metrics['expected_loss']:.2f}  VaR_95={risk_metrics['VaR_95']:.2f}")
            summary_rows.append({"turbine": turbine, "granularity": freq, "experiment": exp_name, **risk_metrics})

        # --- paired incremental loss vs E1 (statistically correct version) ---
        if "E1_raw_tft" in exp_cost_dfs:
            e1_df = exp_cost_dfs["E1_raw_tft"].set_index("group_id")["scenario_loss"]
            for exp_name, cost_df in exp_cost_dfs.items():
                if exp_name == "E1_raw_tft":
                    continue
                m_df = cost_df.set_index("group_id")["scenario_loss"]
                shared_ids = e1_df.index.intersection(m_df.index)
                if len(shared_ids) == 0:
                    continue
                delta = (m_df.loc[shared_ids] - e1_df.loc[shared_ids]).to_numpy()
                delta_metrics = summarize_costs(delta)
                for gid, d in zip(shared_ids, delta):
                    incremental_rows.append({
                        "turbine": turbine, "granularity": freq, "experiment": exp_name,
                        "group_id": gid, "delta_cost": d,
                    })
                print(f"  [\u0394 {exp_name} vs E1] n_paired={len(shared_ids)}  "
                      f"mean_delta={delta_metrics['expected_loss']:.2f}  "
                      f"VaR_95(\u0394)={delta_metrics['VaR_95']:.2f}")

    if not summary_rows:
        print("\nNo results produced.")
        return

    result_df = pd.DataFrame(summary_rows)
    result_df.to_csv(config.SUMMARY_TABLE_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {config.SUMMARY_TABLE_CSV}")

    # --- level-difference savings (VaR(E1) - VaR(E_m), NOT the same as VaR(delta) -- kept for reference) ---
    savings_rows = []
    for (turbine, freq), group in result_df.groupby(["turbine", "granularity"]):
        e1_row = group[group["experiment"] == "E1_raw_tft"]
        if e1_row.empty:
            continue
        e1_cost, e1_var, e1_cvar = e1_row.iloc[0][["expected_loss", "VaR_95", "CVaR_95"]]
        for _, row in group.iterrows():
            if row["experiment"] == "E1_raw_tft":
                continue
            savings_rows.append({
                "turbine": turbine, "granularity": freq, "experiment": row["experiment"],
                "expected_cost_saving_vs_E1": e1_cost - row["expected_loss"],
                "VaR_95_reduction_pct_vs_E1": (e1_var - row["VaR_95"]) / abs(e1_var) * 100 if e1_var else float("nan"),
                "CVaR_95_reduction_pct_vs_E1": (e1_cvar - row["CVaR_95"]) / abs(e1_cvar) * 100 if e1_cvar else float("nan"),
            })
    if savings_rows:
        savings_df = pd.DataFrame(savings_rows)
        savings_path = config.OUTPUT_DIR / "savings_vs_E1_baseline.csv"
        savings_df.to_csv(savings_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {savings_path}  (level-difference VaR/CVaR, see incremental_loss_vs_E1.csv for the paired version)")

    # --- paired incremental loss summary (the statistically correct version) ---
    if incremental_rows:
        incr_df = pd.DataFrame(incremental_rows)
        incr_detail_path = config.OUTPUT_DIR / "incremental_loss_vs_E1_detail.csv"
        incr_df.to_csv(incr_detail_path, index=False, encoding="utf-8-sig")

        incr_summary_rows = []
        for (turbine, freq, exp), group in incr_df.groupby(["turbine", "granularity", "experiment"]):
            m = summarize_costs(group["delta_cost"].to_numpy())
            incr_summary_rows.append({"turbine": turbine, "granularity": freq, "experiment": exp,
                                       "n_paired_scenarios": m["n_samples"],
                                       "mean_delta_cost": m["expected_loss"],
                                       "VaR_95_delta": m["VaR_95"], "CVaR_95_delta": m["CVaR_95"],
                                       "VaR_99_delta": m["VaR_99"], "CVaR_99_delta": m["CVaR_99"]})
        incr_summary_df = pd.DataFrame(incr_summary_rows)
        incr_summary_path = config.OUTPUT_DIR / "incremental_loss_vs_E1.csv"
        incr_summary_df.to_csv(incr_summary_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {incr_summary_path}  (PAIRED per-scenario delta vs E1 -- use this for the paper)")
    else:
        incr_df = pd.DataFrame()

    # must run AFTER incremental_loss_vs_E1.csv is written above, since this
    # function reads that file
    auto_merge_accuracy_and_economics()

    print("\nGenerating figures...")
    fig_dir = config.OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_var_cvar_heatmap(result_df, fig_dir)
    for freq in FREQ_ORDER:
        plot_loss_distribution(RAW_COSTS_DIR, freq, fig_dir)
    plot_incremental_loss_distribution(incr_df, fig_dir)
    plot_loss_vs_spread_scatter(RAW_COSTS_DIR, fig_dir)


if __name__ == "__main__":
    main()
