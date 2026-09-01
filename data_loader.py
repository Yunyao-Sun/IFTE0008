"""
Discover and load pointwise CSVs produced by the inference scripts.
Expected filename pattern: {turbine}_{freq}_{experiment}_pointwise.csv
"""

import re
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

import config

FILENAME_PATTERN = re.compile(
    r"^(?P<turbine>[^_]+)_(?P<freq>\d+min)_(?P<experiment>E\d_[a-zA-Z0-9_]+)_pointwise\.csv$"
)


@dataclass
class PointwiseFile:
    turbine: str
    freq: str
    experiment: str
    path: Path


def discover_pointwise_files(directory: Path = None) -> list[PointwiseFile]:
    directory = directory or config.POINTWISE_DIR
    files = []
    for path in sorted(directory.glob("*_pointwise.csv")):
        m = FILENAME_PATTERN.match(path.name)
        if not m:
            print(f"  [skip] filename does not match expected pattern: {path.name}")
            continue
        files.append(PointwiseFile(
            turbine=m.group("turbine"),
            freq=m.group("freq"),
            experiment=m.group("experiment"),
            path=path,
        ))
    return files


def load_pointwise(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"group_id", "horizon_step", "actual_kw", "pred_p50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def freq_to_minutes(freq: str) -> int:
    return int(freq.replace("min", ""))


def load_price_data(path=None) -> pd.DataFrame:
    """
    Load real Elexon SSP/SBP price CSV (produced by fetch_prices.py).
    Only needed when config.USE_REAL_PRICE = True.
    Expected columns: settlementDate, settlementPeriod, systemSellPrice, systemBuyPrice

    Sorted into true chronological order (by settlementDate then
    settlementPeriod) so that consecutive rows really are consecutive
    30-minute settlement periods -- this matters for sampling realistic
    continuous historical price paths (see imbalance_from_pointwise.py).
    """
    path = path or config.PRICE_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Real price file not found: {path}\n"
            f"Either run fetch_prices.py first, or set config.USE_REAL_PRICE = False "
            f"to use the simplified fixed price instead."
        )
    df = pd.read_csv(path)
    required = {"systemSellPrice", "systemBuyPrice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if "settlementDate" in df.columns and "settlementPeriod" in df.columns:
        df["settlementDate"] = pd.to_datetime(df["settlementDate"])
        df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    return df
