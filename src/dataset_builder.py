"""
dataset_builder.py  ── 修复版
==============================
核心修改：
  1. MAX_SAMPLES_PER_SPLIT=None 支持（不限制样本数）
  2. EN 只在训练集 fit 一次，验证/测试集复用 selected_indices
  3. target 做全局 min-max 归一化（替代 GroupNormalizer）
  4. 特征 X 做全局 StandardScaler（只在训练集 fit）
  5. LSTM arrays 同步归一化
  6. make_tft_dataset 用 TorchNormalizer(identity)
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer

from config import CFG
from src.decomposition import compact_mvmd, modes_to_features
from src.feature_selection import elastic_net_fit, elastic_net_apply


# ---------------------------------------------------------------
# 全局归一化器
# ---------------------------------------------------------------
class _GlobalScaler:
    def __init__(self):
        self.min_ = self.max_ = None
        self._fitted = False

    def fit(self, y: np.ndarray):
        self.min_ = float(np.nanmin(y))
        self.max_ = float(np.nanmax(y))
        self._fitted = True
        print(f"  TargetScaler fit: min={self.min_:.2f} max={self.max_:.2f}")
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        assert self._fitted
        return (y - self.min_) / (self.max_ - self.min_ + 1e-9)

    def inverse_transform(self, y_norm: np.ndarray) -> np.ndarray:
        assert self._fitted
        return y_norm * (self.max_ - self.min_ + 1e-9) + self.min_

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)


TARGET_SCALER  = _GlobalScaler()
FEATURE_SCALER = StandardScaler()
_feat_fitted   = False

EN_SELECTED_INDICES: Optional[np.ndarray] = None
EN_SELECTED_NAMES:   Optional[List[str]]  = None
EN_FEATURE_COUNT:    Optional[int]        = None


def reset_global_state():
    global _feat_fitted, EN_SELECTED_INDICES, EN_SELECTED_NAMES, EN_FEATURE_COUNT
    TARGET_SCALER.__init__()
    FEATURE_SCALER.__init__()
    _feat_fitted = False
    EN_SELECTED_INDICES = EN_SELECTED_NAMES = EN_FEATURE_COUNT = None


# ---------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------
def get_observed_channels(df: pd.DataFrame, include_target: bool) -> List[str]:
    excluded = set(CFG.TIME_COLS)
    if not include_target:
        excluded.add(CFG.TARGET_COL)
    return [col for col in df.columns if col not in excluded]


def _sample_starts(df: pd.DataFrame) -> list:
    n = len(df) - CFG.WINDOW_SIZE - CFG.HORIZON + 1
    starts = list(range(0, max(n, 0), CFG.STEP))
    if CFG.MAX_SAMPLES_PER_SPLIT is not None:
        starts = starts[:CFG.MAX_SAMPLES_PER_SPLIT]
    return starts


def _pad_or_truncate(X: np.ndarray, n: int) -> np.ndarray:
    if X.shape[1] < n:
        return np.column_stack([X, np.zeros((X.shape[0], n - X.shape[1]))])
    return X[:, :n]


# ---------------------------------------------------------------
# build_tft_rows  (修复版)
# ---------------------------------------------------------------
def build_tft_rows(
    df: pd.DataFrame,
    experiment: str,
    split_name: str,
    is_train: bool = False,
    global_features: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    global _feat_fitted, EN_SELECTED_INDICES, EN_SELECTED_NAMES, EN_FEATURE_COUNT

    starts = _sample_starts(df)
    raw_cols  = get_observed_channels(df, include_target=False)
    mvmd_cols = get_observed_channels(df, include_target=True)

    # --- PASS 1: 训练集 fit scaler & EN ---
    if is_train:
        all_X, all_y, feat_ref = [], [], None
        for start in starts:
            enc = df.iloc[start: start + CFG.WINDOW_SIZE]
            if experiment == "E1_raw_tft":
                X = enc[raw_cols].values; feat_ref = raw_cols
            elif experiment == "E2_global_mvmd_tft":
                if global_features is None:
                    raise ValueError("global_features required for E2")
                X = global_features.iloc[start: start + CFG.WINDOW_SIZE].values
                feat_ref = list(global_features.columns)
            elif experiment in ("E3_rolling_mvmd_tft", "E5_rolling_mvmd_en_tft"):
                modes, _ = compact_mvmd(enc[mvmd_cols].values)
                X, feat_ref = modes_to_features(modes, mvmd_cols)
            else:
                raise ValueError(f"Unknown experiment: {experiment}")
            all_X.append(X)
            all_y.append(enc[CFG.TARGET_COL].values)

        all_X_flat = np.vstack(all_X)
        all_y_flat = np.concatenate(all_y)

        TARGET_SCALER.fit(df[CFG.TARGET_COL].values)
        FEATURE_SCALER.fit(all_X_flat)
        _feat_fitted    = True
        EN_FEATURE_COUNT = all_X_flat.shape[1]

        if experiment == "E5_rolling_mvmd_en_tft":
            EN_SELECTED_INDICES, EN_SELECTED_NAMES = elastic_net_fit(
                all_X_flat, all_y_flat, feat_ref or [])
            print(f"  EN selected {len(EN_SELECTED_INDICES)} features")

    # --- PASS 2: 构建行 ---
    print(f"  [{split_name}|{experiment}] building {len(starts)} samples...")
    rows: List[dict] = []
    selected_log: List[str] = []
    feature_count: Optional[int] = None

    for sid, start in enumerate(starts):
        if sid % 300 == 0:
            print(f"    {sid}/{len(starts)}")

        enc = df.iloc[start: start + CFG.WINDOW_SIZE]
        dec = df.iloc[start + CFG.WINDOW_SIZE: start + CFG.WINDOW_SIZE + CFG.HORIZON]

        if experiment == "E1_raw_tft":
            X_enc = enc[raw_cols].values
        elif experiment == "E2_global_mvmd_tft":
            X_enc = global_features.iloc[start: start + CFG.WINDOW_SIZE].values
        elif experiment in ("E3_rolling_mvmd_tft", "E5_rolling_mvmd_en_tft"):
            modes, _ = compact_mvmd(enc[mvmd_cols].values)
            X_enc, _ = modes_to_features(modes, mvmd_cols)
        else:
            raise ValueError(f"Unknown experiment: {experiment}")

        if _feat_fitted:
            X_enc = FEATURE_SCALER.transform(
                _pad_or_truncate(X_enc, EN_FEATURE_COUNT))

        if experiment == "E5_rolling_mvmd_en_tft" and EN_SELECTED_INDICES is not None:
            X_enc = X_enc[:, EN_SELECTED_INDICES]
            selected_log.extend(EN_SELECTED_NAMES or [])

        if feature_count is None:
            feature_count = X_enc.shape[1]
        X_enc = _pad_or_truncate(X_enc, feature_count)

        gid = f"{split_name}_{sid}"

        for t in range(CFG.WINDOW_SIZE):
            t_raw  = float(enc[CFG.TARGET_COL].iloc[t])
            t_norm = float(TARGET_SCALER.transform(np.array([t_raw]))[0])
            row = {"group_id": gid, "time_idx": t,
                   "target": t_norm,
                   "relative_time_idx": float(t), "encoder_flag": 1.0}
            for col in CFG.TIME_COLS:
                if col in enc.columns:
                    row[col] = float(enc[col].iloc[t])
            for j in range(feature_count):
                row[f"x_{j+1}"] = float(X_enc[t, j])
            rows.append(row)

        for h in range(CFG.HORIZON):
            t = CFG.WINDOW_SIZE + h
            t_raw  = float(dec[CFG.TARGET_COL].iloc[h])
            t_norm = float(TARGET_SCALER.transform(np.array([t_raw]))[0])
            row = {"group_id": gid, "time_idx": t,
                   "target": t_norm,
                   "relative_time_idx": float(t), "encoder_flag": 0.0}
            for col in CFG.TIME_COLS:
                if col in dec.columns:
                    row[col] = float(dec[col].iloc[h])
            for j in range(feature_count):
                row[f"x_{j+1}"] = 0.0
            rows.append(row)

    return pd.DataFrame(rows), selected_log


# ---------------------------------------------------------------
# make_tft_dataset  (用 identity normalizer)
# ---------------------------------------------------------------
def make_tft_dataset(tft_df: pd.DataFrame,
                     training_ds: TimeSeriesDataSet = None) -> TimeSeriesDataSet:
    W, H = CFG.WINDOW_SIZE, CFG.HORIZON
    feat_cols  = sorted(c for c in tft_df.columns if c.startswith("x_"))
    time_cols  = [c for c in CFG.TIME_COLS if c in tft_df.columns]
    known_reals   = time_cols + ["relative_time_idx", "encoder_flag"]
    unknown_reals = feat_cols + ["target"]

    if training_ds is None:
        return TimeSeriesDataSet(
            tft_df,
            time_idx              = "time_idx",
            target                = "target",
            group_ids             = ["group_id"],
            min_encoder_length    = W,
            max_encoder_length    = W,
            min_prediction_length = H,
            max_prediction_length = H,
            time_varying_known_reals   = known_reals,
            time_varying_unknown_reals = unknown_reals,
            target_normalizer     = TorchNormalizer(method="identity", center=False),
            allow_missing_timesteps = True,
            add_relative_time_idx   = False,
        )
    return TimeSeriesDataSet.from_dataset(
        training_ds, tft_df, predict=True, stop_randomization=True)


# ---------------------------------------------------------------
# extract_decoder_targets
# ---------------------------------------------------------------
def extract_decoder_targets(tft_rows: pd.DataFrame,
                             inverse: bool = True) -> np.ndarray:
    grouped = (
        tft_rows[tft_rows["encoder_flag"] == 0]
        .sort_values(["group_id", "time_idx"])
        .groupby("group_id")["target"]
        .apply(lambda s: s.values)
    )
    y = np.vstack(grouped.values)
    if inverse:
        y = TARGET_SCALER.inverse_transform(y)
    return y


# ---------------------------------------------------------------
# build_lstm_arrays  (修复版)
# ---------------------------------------------------------------
def build_lstm_arrays(df: pd.DataFrame,
                      split_name: str,
                      is_train: bool = False) -> Tuple[dict, List[str]]:
    global _feat_fitted, EN_SELECTED_INDICES, EN_SELECTED_NAMES, EN_FEATURE_COUNT

    starts    = _sample_starts(df)
    mvmd_cols = get_observed_channels(df, include_target=True)
    Xs, ys, selected_log = [], [], []

    if is_train and EN_SELECTED_INDICES is None:
        all_X, all_y, feat_ref = [], [], None
        for start in starts:
            enc = df.iloc[start: start + CFG.WINDOW_SIZE]
            modes, _ = compact_mvmd(enc[mvmd_cols].values)
            X_all, feat_ref = modes_to_features(modes, mvmd_cols)
            all_X.append(X_all)
            all_y.append(enc[CFG.TARGET_COL].values)

        all_X_flat = np.vstack(all_X)
        all_y_flat = np.concatenate(all_y)

        if not _feat_fitted:
            FEATURE_SCALER.fit(all_X_flat)
            TARGET_SCALER.fit(df[CFG.TARGET_COL].values)
            _feat_fitted = True
        EN_FEATURE_COUNT = all_X_flat.shape[1]
        EN_SELECTED_INDICES, EN_SELECTED_NAMES = elastic_net_fit(
            all_X_flat, all_y_flat, feat_ref or [])
        print(f"  EN selected {len(EN_SELECTED_INDICES)} features")

    print(f"  [{split_name}|E4_lstm] building {len(starts)} samples...")
    for sid, start in enumerate(starts):
        if sid % 300 == 0:
            print(f"    {sid}/{len(starts)}")

        enc = df.iloc[start: start + CFG.WINDOW_SIZE]
        dec = df.iloc[start + CFG.WINDOW_SIZE: start + CFG.WINDOW_SIZE + CFG.HORIZON]

        modes, _ = compact_mvmd(enc[mvmd_cols].values)
        X_all, names_all = modes_to_features(modes, mvmd_cols)

        X_scaled = FEATURE_SCALER.transform(
            _pad_or_truncate(X_all, EN_FEATURE_COUNT or X_all.shape[1]))
        if EN_SELECTED_INDICES is not None:
            X_sel = X_scaled[:, EN_SELECTED_INDICES]
            selected_log.extend(EN_SELECTED_NAMES or [])
        else:
            X_sel = X_scaled
        X_sel = _pad_or_truncate(X_sel, CFG.EN_TOP_N)

        y_raw  = dec[CFG.TARGET_COL].values.astype(np.float32)
        y_norm = TARGET_SCALER.transform(y_raw).astype(np.float32)

        Xs.append(X_sel.astype(np.float32))
        ys.append(y_norm)

    return {"X": np.asarray(Xs), "y": np.asarray(ys)}, selected_log


def inverse_target(y_norm: np.ndarray) -> np.ndarray:
    return TARGET_SCALER.inverse_transform(y_norm)
