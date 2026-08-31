"""Preprocessing for Penmanshiel SCADA data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col not in ["date_time", "source_file"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _merge_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in df.columns if col not in ["date_time", "source_file"]]
    return df.groupby("date_time", as_index=False)[numeric_cols].mean()


def _build_blade_angle(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    blade_cols = [col for col in ["blade_angle_a", "blade_angle_b", "blade_angle_c"] if col in out.columns]
    if blade_cols:
        out["blade_angle_mean"] = out[blade_cols].mean(axis=1)

    if "blade_angle_existing" in out.columns:
        if "blade_angle_mean" in out.columns:
            out["blade_angle_mean"] = out["blade_angle_mean"].fillna(out["blade_angle_existing"])
        else:
            out["blade_angle_mean"] = out["blade_angle_existing"]

    drop_cols = blade_cols + ["blade_angle_existing"]
    return out.drop(columns=drop_cols, errors="ignore")


def _apply_physical_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if CFG.TARGET_COL in out.columns:
        out.loc[(out[CFG.TARGET_COL] < 0) | (out[CFG.TARGET_COL] > CFG.RATED_POWER * 1.2), CFG.TARGET_COL] = np.nan

    if "wind_speed" in out.columns:
        out.loc[(out["wind_speed"] < 0) | (out["wind_speed"] > 40), "wind_speed"] = np.nan

    if "wind_speed_std" in out.columns:
        out.loc[(out["wind_speed_std"] < 0) | (out["wind_speed_std"] > 20), "wind_speed_std"] = np.nan

    for col in ["wind_direction", "nacelle_position", "blade_angle_mean"]:
        if col in out.columns:
            out.loc[(out[col] < -5) | (out[col] > 365), col] = np.nan

    for col in ["nacelle_temp", "gear_oil_temp", "gear_oil_inlet_temp"]:
        if col in out.columns:
            out.loc[(out[col] < -40) | (out[col] > 110), col] = np.nan

    if "rotor_speed" in out.columns:
        out.loc[(out["rotor_speed"] < 0) | (out["rotor_speed"] > 40), "rotor_speed"] = np.nan

    if "grid_voltage" in out.columns:
        out.loc[out["grid_voltage"] < 0, "grid_voltage"] = np.nan

    if "grid_frequency" in out.columns:
        out.loc[(out["grid_frequency"] < 45) | (out["grid_frequency"] > 55), "grid_frequency"] = np.nan

    return out


def _add_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["wind_direction", "nacelle_position", "blade_angle_mean"]:
        if col in out.columns:
            rad = np.deg2rad(out[col] % 360)
            out[f"{col}_sin"] = np.sin(rad)
            out[f"{col}_cos"] = np.cos(rad)
            if col in ["wind_direction", "nacelle_position"]:
                out = out.drop(columns=[col])
    return out


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out.index.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out.index.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * out.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out.index.month / 12)
    return out


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "wind_speed" in out.columns and "wind_speed_std" in out.columns:
        out["turbulence_intensity"] = (out["wind_speed_std"] / (out["wind_speed"].abs() + 1e-3)).clip(0, 5)
    return out


def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _to_numeric(df_raw)
    df = _merge_duplicate_timestamps(df)
    df = _build_blade_angle(df)
    df = _apply_physical_filters(df)

    df = _add_angle_features(df)

    df = df.set_index("date_time").sort_index()
    df = df.resample(CFG.FREQ).mean()
    df = df.interpolate(method="time", limit=CFG.SHORT_GAP_LIMIT, limit_direction="both")

    df = _add_derived_features(df)
    df = _add_time_features(df)

    if CFG.TARGET_COL in df.columns:
        df = df.dropna(subset=[CFG.TARGET_COL])

    missing_ratio = df.isna().mean().sort_values(ascending=False)
    missing_ratio.to_csv(CFG.RESULTS_DIR / "missing_ratio_after_preprocessing.csv")
    print("Missing ratio after preprocessing:")
    print(missing_ratio)

    cols_to_drop = missing_ratio[missing_ratio > CFG.MAX_MISSING_RATIO].index.tolist()
    if cols_to_drop:
        print(f"Dropping columns with more than {CFG.MAX_MISSING_RATIO:.0%} missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    df = df.interpolate(method="time", limit=CFG.SHORT_GAP_LIMIT, limit_direction="both")
    df = df.dropna()

    output_path = CFG.RESULTS_DIR / "cleaned_data.csv"
    df.to_csv(output_path)
    print(f"Cleaned data saved to {output_path}")
    print(f"Cleaned shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * CFG.TRAIN_RATIO)
    val_end = int(n * (CFG.TRAIN_RATIO + CFG.VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print("Data split:")
    print(f"Train: {train_df.index.min()} -> {train_df.index.max()} | n={len(train_df)}")
    print(f"Val:   {val_df.index.min()} -> {val_df.index.max()} | n={len(val_df)}")
    print(f"Test:  {test_df.index.min()} -> {test_df.index.max()} | n={len(test_df)}")
    return train_df, val_df, test_df
