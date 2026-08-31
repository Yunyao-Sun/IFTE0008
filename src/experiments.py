"""Experiment runners for five ablation settings."""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet

from config import CFG
from src.dataset_builder import build_lstm_arrays, build_tft_rows, extract_decoder_targets, get_observed_channels, inverse_target
from src.decomposition import global_mvmd_features
from src.metrics import compute_metrics, save_experiment_result
from src.models import predict_lstm, predict_tft, train_lstm, train_tft


def _build_tft_dataset(train_rows: pd.DataFrame, val_rows: pd.DataFrame, test_rows: pd.DataFrame):
    x_cols = sorted(
        [col for col in train_rows.columns if col.startswith("x_")],
        key=lambda col: int(col.split("_")[1]),
    )
    known_cols = [col for col in CFG.TIME_COLS if col in train_rows.columns]

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
    val_ds = TimeSeriesDataSet.from_dataset(training_ds, val_rows, stop_randomization=True)
    test_ds = TimeSeriesDataSet.from_dataset(training_ds, test_rows, stop_randomization=True)
    return training_ds, val_ds, test_ds


def _run_tft_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exp_name: str,
    global_train: Optional[pd.DataFrame] = None,
    global_val: Optional[pd.DataFrame] = None,
    global_test: Optional[pd.DataFrame] = None,
) -> dict:
    start_time = time.time()

    # 修复：build_tft_rows(df, experiment, split_name, is_train, global_features)
    # 之前按位置传参，global_train/val/test 落进了 is_train 的位置，
    # 导致训练集该做的 scaler/EN 拟合被跳过（E1/E3/E5），
    # E2 时 is_train 收到一个非空 DataFrame 直接触发
    # "truth value of a DataFrame is ambiguous" 报错。
    # 用关键字参数明确传递，训练集 is_train=True，验证/测试集 is_train=False。
    train_rows, train_log = build_tft_rows(
        train_df, exp_name, "train", is_train=True, global_features=global_train)
    val_rows, val_log = build_tft_rows(
        val_df, exp_name, "val", is_train=False, global_features=global_val)
    test_rows, test_log = build_tft_rows(
        test_df, exp_name, "test", is_train=False, global_features=global_test)

    if train_log or val_log or test_log:
        selected = pd.Series(train_log + val_log + test_log).value_counts()
        selected.to_csv(CFG.RESULTS_DIR / f"{exp_name}_selected_feature_counts.csv")

    training_ds, val_ds, test_ds = _build_tft_dataset(train_rows, val_rows, test_rows)
    model = train_tft(training_ds, val_ds, exp_name)
    pred_q = predict_tft(model, test_ds)
    y_true = extract_decoder_targets(test_rows)

    metrics_df = compute_metrics(y_true, pred_q)
    metrics_df["experiment"] = exp_name
    metrics_df["runtime_sec"] = time.time() - start_time
    save_experiment_result(metrics_df, exp_name)
    return {"metrics": metrics_df, "model": model}


def run_e1(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    return _run_tft_experiment(train_df, val_df, test_df, "E1_raw_tft")


def run_e2(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    combined = pd.concat([train_df, val_df, test_df], axis=0)
    channels = get_observed_channels(combined, include_target=True)
    all_global = global_mvmd_features(combined, channels)

    n_train = len(train_df)
    n_val = len(val_df)
    global_train = all_global.iloc[:n_train]
    global_val = all_global.iloc[n_train:n_train + n_val]
    global_test = all_global.iloc[n_train + n_val:]

    return _run_tft_experiment(
        train_df,
        val_df,
        test_df,
        "E2_global_mvmd_tft",
        global_train,
        global_val,
        global_test,
    )


def run_e3(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    return _run_tft_experiment(train_df, val_df, test_df, "E3_rolling_mvmd_tft")


def run_e4(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    start_time = time.time()

    # 修复：build_lstm_arrays(df, split_name, is_train) 之前调用时没传
    # is_train，训练集该做的 Elastic Net 特征选择拟合被跳过。
    train_samples, train_log = build_lstm_arrays(train_df, "train", is_train=True)
    val_samples, val_log = build_lstm_arrays(val_df, "val", is_train=False)
    test_samples, test_log = build_lstm_arrays(test_df, "test", is_train=False)

    if train_log or val_log or test_log:
        selected = pd.Series(train_log + val_log + test_log).value_counts()
        selected.to_csv(CFG.RESULTS_DIR / "E4_rolling_mvmd_en_lstm_selected_feature_counts.csv")

    model = train_lstm(train_samples, val_samples, "E4_rolling_mvmd_en_lstm")
    pred_q = predict_lstm(model, test_samples)

    # 修复：test_samples["y"] 是 build_lstm_arrays 里存的归一化 [0,1] 真值，
    # 而 pred_q（predict_lstm 默认 inverse=True）是反归一化后的原始功率(kW)。
    # 两者单位不一致会导致 RMSE/PICP/MAPE 全部失真（PICP 趋近于0，
    # RMSE 虚高），必须先把真值也反归一化到同一个kW尺度再比较。
    y_true = inverse_target(test_samples["y"])
    metrics_df = compute_metrics(y_true, pred_q)
    metrics_df["experiment"] = "E4_rolling_mvmd_en_lstm"
    metrics_df["runtime_sec"] = time.time() - start_time
    save_experiment_result(metrics_df, "E4_rolling_mvmd_en_lstm")
    return {"metrics": metrics_df, "model": model}


def run_e5(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, selector_e4=None) -> dict:
    return _run_tft_experiment(train_df, val_df, test_df, "E5_rolling_mvmd_en_tft")
