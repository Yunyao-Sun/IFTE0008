from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# The only thing you need to check before running this file:
# this path must point to the current turbine+granularity's
# all_5_ablation_summary.csv (the one matching whatever
# TURBINE_ID / FREQ you just set in config.py)
# ---------------------------------------------------------------
SUMMARY_CSV_PATH = "results/all_5_ablation_summary.csv"
TOLERANCE = 5.0

print("=== run_all.py: inference only, no training. ===")
print("=== If you see epoch progress bars, stop immediately with Ctrl+C. ===\n")

from config import CFG
from src.data_loader import load_raw
from src.preprocessing import preprocess, split_data
from src.dataset_builder import (
    build_tft_rows,
    build_lstm_arrays,
    extract_decoder_targets,
    inverse_target,
    reset_global_state,
    get_observed_channels,
)
from src.decomposition import global_mvmd_features
from src.metrics import compute_metrics
from src.models import predict_tft, predict_lstm, QuantileLSTM, set_global_normalizer
from src.dataset_builder import TARGET_SCALER

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

EXPERIMENTS = [
    "E1_raw_tft",
    "E2_global_mvmd_tft",
    "E3_rolling_mvmd_tft",
    "E4_rolling_mvmd_en_lstm",
    "E5_rolling_mvmd_en_tft",
]

RMSE_COLUMN_CANDIDATES = ["RMSE_mean", "RMSE", "RMSE(kW)"]


def build_tft_dataset(train_rows: pd.DataFrame, test_rows: pd.DataFrame):
    x_cols = sorted(
        [c for c in train_rows.columns if c.startswith("x_")],
        key=lambda c: int(c.split("_")[1]),
    )
    known_cols = [c for c in CFG.TIME_COLS if c in train_rows.columns]

    training_ds = TimeSeriesDataSet(
        train_rows,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        min_encoder_length=CFG.WINDOW_SIZE,
        max_encoder_length=CFG.WINDOW_SIZE,
        min_prediction_length=CFG.HORIZON,
        max_prediction_length=CFG.HORIZON,
        time_varying_known_reals=["relative_time_idx", "encoder_flag"] + known_cols,
        time_varying_unknown_reals=["target"] + x_cols,
        target_normalizer=None,
        allow_missing_timesteps=False,
    )
    test_ds = TimeSeriesDataSet.from_dataset(training_ds, test_rows, stop_randomization=True)
    return training_ds, test_ds


def prepare_data_for_experiment(exp_name: str, train_df, val_df, test_df):
    reset_global_state()

    if exp_name == "E4_rolling_mvmd_en_lstm":
        train_samples, _ = build_lstm_arrays(train_df, "train", is_train=True)
        # same GLOBAL_NORM sync issue as the TFT branch below
        set_global_normalizer(TARGET_SCALER)
        test_samples, _ = build_lstm_arrays(test_df, "test", is_train=False)
        return {"kind": "lstm", "train_samples": train_samples, "test_samples": test_samples}

    global_train = global_val = global_test = None
    if exp_name == "E2_global_mvmd_tft":
        combined = pd.concat([train_df, val_df, test_df], axis=0)
        channels = get_observed_channels(combined, include_target=True)
        all_global = global_mvmd_features(combined, channels)
        n_train, n_val = len(train_df), len(val_df)
        global_train = all_global.iloc[:n_train]
        global_val = all_global.iloc[n_train:n_train + n_val]
        global_test = all_global.iloc[n_train + n_val:]

    train_rows, _ = build_tft_rows(
        train_df, exp_name, "train", is_train=True, global_features=global_train)
    # TARGET_SCALER just got fit() inside build_tft_rows above (is_train=True).
    # predict_tft()/predict_lstm() actually call models.GLOBAL_NORM.inverse_transform(),
    # a SEPARATE object from TARGET_SCALER, so it must be synced explicitly here,
    # exactly like the original main.py does after fitting TARGET_SCALER.
    set_global_normalizer(TARGET_SCALER)

    test_rows, _ = build_tft_rows(
        test_df, exp_name, "test", is_train=False, global_features=global_test)

    training_ds, test_ds = build_tft_dataset(train_rows, test_rows)
    return {"kind": "tft", "test_rows": test_rows, "training_ds": training_ds, "test_ds": test_ds}


def try_checkpoint(ckpt_path: str, prepared: dict):
    if prepared["kind"] == "lstm":
        input_size = prepared["train_samples"]["X"].shape[2]
        model = QuantileLSTM.load_from_checkpoint(ckpt_path, input_size=input_size)
        pred_q = predict_lstm(model, prepared["test_samples"])
        y_true = inverse_target(prepared["test_samples"]["y"])
    else:
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
        pred_q = predict_tft(model, prepared["test_ds"])
        y_true = extract_decoder_targets(prepared["test_rows"])

    metrics_df = compute_metrics(y_true, pred_q)
    rmse = float(metrics_df["RMSE"].mean()) if "RMSE" in metrics_df.columns else float("nan")
    return rmse, y_true, pred_q


def save_pointwise(y_true: np.ndarray, pred_q: np.ndarray, exp_name: str):
    n_samples, horizon = y_true.shape[0], y_true.shape[1]
    rows = []
    has_quantiles = pred_q.ndim == 3
    for i in range(n_samples):
        for h in range(horizon):
            row = {"group_id": i, "horizon_step": h, "actual_kw": float(y_true[i, h])}
            if has_quantiles:
                row["pred_p10"] = float(pred_q[i, h, 0])
                row["pred_p50"] = float(pred_q[i, h, 1])
                row["pred_p90"] = float(pred_q[i, h, 2])
            else:
                row["pred_p50"] = float(pred_q[i, h])
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_name = f"{CFG.TURBINE_ID}_{CFG.FREQ}_{exp_name}_pointwise.csv"
    out_path = CFG.RESULTS_DIR / out_name
    out_df.to_csv(out_path, index=False)
    return out_path


def get_expected_rmse(summary_df: pd.DataFrame, exp_name: str):
    row = summary_df[summary_df["experiment"] == exp_name]
    if row.empty:
        return None
    for col in RMSE_COLUMN_CANDIDATES:
        if col in row.columns:
            return float(row.iloc[0][col])
    return None


def run_one_experiment(exp_name: str, expected_rmse: float, prepared_cache: dict,
                        train_df, val_df, test_df) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Experiment: {exp_name}   Expected RMSE: {expected_rmse:.2f}")
    print("=" * 60)

    if exp_name not in prepared_cache:
        print("  Rebuilding scalers/features for this experiment type...")
        prepared_cache[exp_name] = prepare_data_for_experiment(exp_name, train_df, val_df, test_df)
    prepared = prepared_cache[exp_name]

    ckpt_dir = CFG.CHECKPOINT_DIR / exp_name
    ckpt_files = sorted(glob.glob(str(ckpt_dir / "*.ckpt")))
    if not ckpt_files:
        print(f"  ERROR: no .ckpt files in {ckpt_dir}, skipping")
        return {"status": "no_checkpoints"}

    best_match, best_diff = None, float("inf")
    results_cache = {}

    for ckpt_path in ckpt_files:
        name = os.path.basename(ckpt_path)
        try:
            rmse, y_true, pred_q = try_checkpoint(ckpt_path, prepared)
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
            continue
        diff = abs(rmse - expected_rmse)
        print(f"  {name}: inference RMSE={rmse:.2f}  diff={diff:.2f}")
        results_cache[ckpt_path] = (y_true, pred_q)
        if diff < best_diff:
            best_diff, best_match = diff, ckpt_path

    if best_match is None:
        return {"status": "all_failed"}

    status = "ok" if best_diff <= TOLERANCE else "low_confidence"
    print(f"  -> best match: {os.path.basename(best_match)}  diff={best_diff:.2f}  status={status}")

    y_true, pred_q = results_cache[best_match]
    out_path = save_pointwise(y_true, pred_q, exp_name)
    print(f"  saved: {out_path}")

    return {"status": status, "checkpoint": os.path.basename(best_match),
            "rmse_diff": best_diff, "output_file": str(out_path)}


def main():
    print(f"Turbine: {CFG.TURBINE_ID}   Freq: {CFG.FREQ}")
    print(f"Reading expected RMSE values from: {SUMMARY_CSV_PATH}")

    if not os.path.exists(SUMMARY_CSV_PATH):
        print(f"\nERROR: file not found: {SUMMARY_CSV_PATH}")
        print("Edit SUMMARY_CSV_PATH near the top of this file to point to the "
              "correct summary CSV for this turbine+granularity, then run again.")
        return

    summary_df = pd.read_csv(SUMMARY_CSV_PATH)

    print("\nLoading and preprocessing data (once, reused across all 5 experiments)...")
    df_raw = load_raw()
    df_clean = preprocess(df_raw)
    train_df, val_df, test_df = split_data(df_clean)

    prepared_cache: dict = {}
    log = []

    for exp_name in EXPERIMENTS:
        expected_rmse = get_expected_rmse(summary_df, exp_name)
        if expected_rmse is None:
            print(f"\n{exp_name}: not found in summary CSV, skipping")
            log.append({"experiment": exp_name, "status": "not_in_summary"})
            continue
        result = run_one_experiment(exp_name, expected_rmse, prepared_cache,
                                     train_df, val_df, test_df)
        result["experiment"] = exp_name
        log.append(result)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    log_df = pd.DataFrame(log)
    print(log_df.to_string(index=False))

    log_path = CFG.RESULTS_DIR / f"{CFG.TURBINE_ID}_{CFG.FREQ}_inference_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\nLog saved to: {log_path}")

    low_conf = [r for r in log if r.get("status") == "low_confidence"]
    if low_conf:
        print(f"\nWARNING: {len(low_conf)} experiment(s) had RMSE diff above tolerance "
              f"({TOLERANCE}). Their matched checkpoint may be wrong. Review the log above.")


if __name__ == "__main__":
    main()
