"""
Merge System Price and MID into one aligned table, joined on
settlementDate + settlementPeriod. Keeps startTime so downstream code can
verify that a block of settlement periods is genuinely chronologically
contiguous (30 min apart) before using it as an "8-hour historical price
path" -- an inner join or missing days can otherwise silently create gaps.

Can be run standalone:
    python build_combined_price_table.py
or called from main.py's auto-build step via build_combined_price_table().
"""

import pandas as pd
from pathlib import Path

SYSTEM_PRICE_CSV = Path("inputs/elexon_system_price_2020_2022.csv")
MID_CSV = Path("inputs/elexon_mid_2020_2022.csv")
OUTPUT_CSV = Path("inputs/real_price_table_2020_2022.csv")


def build_combined_price_table(system_price_csv: Path = SYSTEM_PRICE_CSV,
                                mid_csv: Path = MID_CSV,
                                output_csv: Path = OUTPUT_CSV) -> pd.DataFrame:
    if not Path(system_price_csv).exists():
        raise FileNotFoundError(f"{system_price_csv} not found.")
    if not Path(mid_csv).exists():
        raise FileNotFoundError(f"{mid_csv} not found. Run fetch_mid.py first.")

    sys_df = pd.read_csv(system_price_csv)
    mid_df = pd.read_csv(mid_csv)

    # GB single cash-out pricing: systemSellPrice == systemBuyPrice, use either.
    sys_df = sys_df.rename(columns={"systemSellPrice": "systemPrice"})
    sys_df = sys_df[["settlementDate", "settlementPeriod", "systemPrice"]]

    sys_df["settlementDate"] = pd.to_datetime(sys_df["settlementDate"]).dt.date.astype(str)
    mid_df["settlementDate"] = pd.to_datetime(mid_df["settlementDate"]).dt.date.astype(str)

    mid_cols = ["settlementDate", "settlementPeriod", "marketIndexPrice"]
    if "startTime" in mid_df.columns:
        mid_cols.append("startTime")

    merged = pd.merge(sys_df, mid_df[mid_cols],
                       on=["settlementDate", "settlementPeriod"], how="inner")

    n_sys, n_merged = len(sys_df), len(merged)
    print(f"System Price rows: {n_sys}")
    print(f"MID rows: {len(mid_df)}")
    print(f"Matched rows (inner join): {n_merged}")
    if n_merged < n_sys * 0.9:
        print(f"WARNING: only {n_merged/n_sys*100:.1f}% of System Price rows found a matching "
              f"MID record. Check settlementDate format / coverage before proceeding.")

    if "startTime" not in merged.columns:
        # Fall back to reconstructing an approximate timestamp from
        # settlementDate + settlementPeriod (each period is 30 min,
        # period 1 = 00:00) so the continuity check downstream still works
        # even if the API didn't return startTime for some reason.
        merged["startTime"] = (
            pd.to_datetime(merged["settlementDate"])
            + pd.to_timedelta((merged["settlementPeriod"] - 1) * 30, unit="min")
        )
    else:
        merged["startTime"] = pd.to_datetime(merged["startTime"], utc=True, errors="coerce")

    merged["settlementDate"] = pd.to_datetime(merged["settlementDate"])
    merged = merged.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)

    price_gap = merged["marketIndexPrice"] - merged["systemPrice"]
    print(f"\nP_MID - P_system stats (this is the spread the cost formula depends on):")
    print(f"  mean = {price_gap.mean():.2f}   std = {price_gap.std():.2f}   "
          f"min = {price_gap.min():.2f}   max = {price_gap.max():.2f}")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {output_csv}")
    return merged


if __name__ == "__main__":
    build_combined_price_table()
