"""Data loading utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from config import CFG


TIME_CANDIDATES = [
    "# Date and time",
    "Date and time",
    "Timestamp",
    "Time",
    "Date",
    "Datetime",
]

COLUMN_ALIASES = {
    "wind_speed": [
        "Wind speed (m/s)",
        "Wind speed",
        "Nacelle wind speed (m/s)",
    ],
    "wind_speed_std": [
        "Wind speed, Standard deviation (m/s)",
        "Wind speed standard deviation (m/s)",
        "Wind speed std",
    ],
    "wind_direction": [
        "Wind direction (°)",
        "Wind direction (Â°)",
        "Wind direction",
    ],
    "nacelle_position": [
        "Nacelle position (°)",
        "Nacelle position (Â°)",
        "Nacelle position",
    ],
    "nacelle_temp": [
        "Nacelle ambient temperature (°C)",
        "Nacelle ambient temperature (Â°C)",
        "Nacelle ambient temperature",
        "Ambient temperature (°C)",
    ],
    "gear_oil_temp": [
        "Gear oil temperature (°C)",
        "Gear oil temperature (Â°C)",
        "Gear oil temperature",
    ],
    "gear_oil_inlet_temp": [
        "Gear oil inlet temperature (°C)",
        "Gear oil inlet temperature (Â°C)",
        "Gear oil inlet temperature",
    ],
    "rotor_speed": [
        "Rotor speed (RPM)",
        "Rotor speed",
    ],
    "blade_angle_a": [
        "Blade angle (pitch position) A (°)",
        "Blade angle (pitch position) A (Â°)",
        "Blade angle A",
    ],
    "blade_angle_b": [
        "Blade angle (pitch position) B (°)",
        "Blade angle (pitch position) B (Â°)",
        "Blade angle B",
    ],
    "blade_angle_c": [
        "Blade angle (pitch position) C (°)",
        "Blade angle (pitch position) C (Â°)",
        "Blade angle C",
    ],
    "blade_angle_existing": [
        "Blade angle (pitch position) (°)",
        "Blade angle (pitch position) (Â°)",
        "Blade angle (pitch position)",
        "Average of Blade angle",
        "Average of blade angle",
    ],
    "grid_voltage": [
        "Grid voltage (V)",
        "Grid voltage",
    ],
    "grid_frequency": [
        "Grid frequency (Hz)",
        "Grid frequency",
    ],
}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {normalize_name(col): col for col in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def discover_data_files() -> list[Path]:
    files: list[Path] = []
    for pattern in CFG.DATA_PATTERNS:
        files.extend(CFG.DATA_DIR.glob(pattern))
    return sorted(set(files))


def read_one_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_raw() -> pd.DataFrame:
    files = discover_data_files()
    if not files:
        raise FileNotFoundError(
            f"No data files found in {CFG.DATA_DIR}. Put turbine files in this directory."
        )

    frames = []
    for path in files:
        print(f"Reading {path.name}")
        raw = read_one_file(path)

        time_col = find_column(raw.columns, TIME_CANDIDATES)
        if time_col is None:
            raise ValueError(f"No time column found in {path.name}. Columns: {list(raw.columns)}")

        rename_map = {time_col: "date_time"}

        target_col = find_column(raw.columns, CFG.REQUIRED_TARGET_NAMES)
        if target_col is not None:
            rename_map[target_col] = CFG.TARGET_COL

        for canonical, aliases in COLUMN_ALIASES.items():
            found = find_column(raw.columns, aliases)
            if found is not None:
                rename_map[found] = canonical

        df = raw.rename(columns=rename_map)
        keep_cols = ["date_time", CFG.TARGET_COL] + list(COLUMN_ALIASES.keys())
        keep_cols = [col for col in keep_cols if col in df.columns]
        df = df[keep_cols].copy()
        df["source_file"] = path.name
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["date_time"] = pd.to_datetime(out["date_time"], errors="coerce")
    out = out.dropna(subset=["date_time"])
    out = out.sort_values("date_time").reset_index(drop=True)

    print(f"Loaded shape: {out.shape}")
    print(f"Loaded columns: {list(out.columns)}")
    return out
