"""Evaluation metrics and result saving."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config import CFG


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    error = y_true - y_pred
    return float(np.mean(np.maximum(q * error, (q - 1) * error)))


def compute_metrics(y_true: np.ndarray, pred_q: np.ndarray) -> pd.DataFrame:
    q_index = {q: i for i, q in enumerate(CFG.TFT_QUANTILES)}
    p50 = pred_q[:, :, q_index[0.5]]

    # MAPE 过滤阈值：只对功率 > 5% 额定容量的时间步计算
    # 避免低功率/停机段（接近0）导致 MAPE 爆炸
    mape_threshold = CFG.RATED_POWER * getattr(CFG, "MAPE_THRESHOLD", 0.05)

    rows = []
    for h in range(y_true.shape[1]):
        yt = y_true[:, h]
        yp = p50[:, h]

        # MAPE：只计算高于阈值的时间步
        mape_mask = yt > mape_threshold
        if mape_mask.sum() > 0:
            mape = float(
                np.mean(
                    np.abs((yt[mape_mask] - yp[mape_mask])
                           / (np.abs(yt[mape_mask]) + 1e-6))
                ) * 100
            )
        else:
            mape = float("nan")

        row = {
            "horizon":      h + 1,
            "RMSE":         float(np.sqrt(np.mean((yt - yp) ** 2))),
            "MAE":          float(np.mean(np.abs(yt - yp))),
            "MAPE":         mape,
            "Pinball_P10":  pinball_loss(yt, pred_q[:, h, q_index[0.1]], 0.1),
            "Pinball_P50":  pinball_loss(yt, pred_q[:, h, q_index[0.5]], 0.5),
            "Pinball_P90":  pinball_loss(yt, pred_q[:, h, q_index[0.9]], 0.9),
        }

        lo = pred_q[:, h, q_index[0.1]]
        hi = pred_q[:, h, q_index[0.9]]
        row["PICP_80"]  = float(np.mean((yt >= lo) & (yt <= hi)))
        row["PINAW_80"] = float(
            np.mean(hi - lo) / (np.max(yt) - np.min(yt) + 1e-8)
        )
        rows.append(row)

    return pd.DataFrame(rows)


def save_experiment_result(metrics_df: pd.DataFrame, exp_name: str) -> None:
    path = CFG.RESULTS_DIR / f"{exp_name}_metrics_by_horizon.csv"
    metrics_df.to_csv(path, index=False)
    print(f"Saved metrics to {path}")


def save_all_summary(all_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, df in all_results.items():
        temp = df.copy()
        temp["experiment"] = name
        frames.append(temp)

    all_by_horizon = pd.concat(frames, ignore_index=True)
    all_by_horizon.to_csv(
        CFG.RESULTS_DIR / "all_experiments_by_horizon.csv", index=False
    )

    summary = (
        all_by_horizon.groupby("experiment")
        .agg(
            RMSE_mean      =("RMSE",        "mean"),
            MAE_mean       =("MAE",         "mean"),
            MAPE_mean      =("MAPE",        "mean"),
            Pinball_P50_mean=("Pinball_P50","mean"),
            PICP_80_mean   =("PICP_80",     "mean"),
            PINAW_80_mean  =("PINAW_80",    "mean"),
        )
        .reset_index()
        .sort_values("RMSE_mean")
    )
    summary.to_csv(CFG.RESULTS_DIR / "all_5_ablation_summary.csv", index=False)
    print(summary)
    return summary
