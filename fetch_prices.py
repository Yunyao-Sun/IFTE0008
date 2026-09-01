"""
Standalone script to fetch 2020-2022 GB settlement system prices (SSP/SBP)
from Elexon's public BMRS API. No API key needed, no config.py dependency.

Usage:
    pip install requests pandas
    python fetch_prices.py

Output:
    elexon_system_price_2020_2022.csv (in the same folder as this script)
"""

import requests
import pandas as pd
import time
from datetime import date, timedelta

START_DATE = date(2020, 1, 1)
END_DATE = date(2022, 12, 31)
OUTPUT_FILE = "elexon_system_price_2020_2022.csv"

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{}"
KEEP_COLUMNS = [
    "settlementDate",
    "settlementPeriod",
    "systemSellPrice",
    "systemBuyPrice",
    "netImbalanceVolume",
]
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2
REQUEST_TIMEOUT = 15


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def fetch_one_day(day: date):
    url = BASE_URL.format(day.isoformat())
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params={"format": "json"}, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                records = resp.json().get("data", [])
                if not records:
                    return None
                df = pd.DataFrame(records)
                cols = [c for c in KEEP_COLUMNS if c in df.columns]
                return df[cols]
            else:
                print(f"  [attempt {attempt}] {day} status={resp.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"  [attempt {attempt}] {day} error={e}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_WAIT_SECONDS)
    print(f"  [failed] {day} gave up after {MAX_RETRIES} attempts")
    return None


def fetch_and_save_prices(start_date: date = START_DATE, end_date: date = END_DATE,
                           output_file: str = OUTPUT_FILE) -> pd.DataFrame:
    all_days = list(daterange(start_date, end_date))
    total = len(all_days)
    print(f"Fetching {start_date} to {end_date}, {total} days total")

    frames = []
    failed_days = []

    for i, day in enumerate(all_days, start=1):
        df_day = fetch_one_day(day)
        if df_day is not None:
            frames.append(df_day)
        else:
            failed_days.append(day)

        if i % 50 == 0 or i == total:
            print(f"Progress: {i}/{total} ({i/total*100:.1f}%)")

    if not frames:
        print("No data retrieved at all, check network connection")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    result.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\nDone: {len(result)} settlement period records")
    print(f"Saved to: {output_file}")

    if failed_days:
        print(f"\n{len(failed_days)} day(s) failed:")
        for d in failed_days:
            print(f"  - {d}")

    return result


if __name__ == "__main__":
    fetch_and_save_prices()
