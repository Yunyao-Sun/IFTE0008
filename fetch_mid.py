"""
Fetch 2020-2022 GB Market Index Data (MID) from Elexon's public BMRS API.
No API key needed. This is a price series INDEPENDENT of System Price
(SSP/SBP), used as the reference market price in
L_t = (E_t - E_hat_t) * (MID_t - SystemPrice_t).

Can be run standalone:
    python fetch_mid.py
or called from main.py's auto-download step via fetch_and_save_mid().

Output columns: settlementDate, settlementPeriod, startTime, marketIndexPrice
marketIndexPrice is the VOLUME-WEIGHTED average across all reporting data
providers (e.g. APX, N2EX) for each settlement period:
    P_MID = sum(price_j * volume_j) / sum(volume_j)
per Elexon's own definition of how the overall Market Index Price is formed
from individual MIDP submissions.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

START_DATE = date(2020, 1, 1)
END_DATE = date(2022, 12, 31)
OUTPUT_FILE = "inputs/elexon_mid_2020_2022.csv"

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
CHUNK_DAYS = 30  # initial guess for request size; automatically split smaller
                  # if Elexon rejects it as exceeding the max allowed range
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3

PRICE_FIELD_CANDIDATES = ["price"]
VOLUME_FIELD_CANDIDATES = ["volume"]
DATE_FIELD_CANDIDATES = ["settlementDate"]
PERIOD_FIELD_CANDIDATES = ["settlementPeriod"]
START_TIME_FIELD_CANDIDATES = ["startTime"]
PROVIDER_FIELD_CANDIDATES = ["dataProvider"]


def pick_field(record: dict, candidates: list):
    for c in candidates:
        if c in record:
            return record[c]
    return None


def daterange_chunks(start: date, end: date, chunk_days: int):
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def _rfc3339(d: date, end_of_day: bool = False) -> str:
    # Elexon docs example format: 2022-06-01T00:00Z
    t = "23:59:59" if end_of_day else "00:00:00"
    return f"{d.isoformat()}T{t}Z"


def fetch_range(start: date, end: date, min_days: int = 1):
    """
    Fetch [start, end] inclusive. If Elexon rejects the range as too long
    (detected from the error response), automatically split it in half and
    retry each half -- this discovers a safe request size on its own,
    instead of guessing a fixed CHUNK_DAYS that might still be wrong.
    """
    params = {
        "from": _rfc3339(start, end_of_day=False),
        "to": _rfc3339(end, end_of_day=True),
        "format": "json",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                payload = resp.json()
                return payload.get("data", payload if isinstance(payload, list) else [])

            body = resp.text[:300]
            range_too_long = resp.status_code == 400 and "must not exceed" in body.lower()
            if range_too_long:
                if start >= end or (end - start).days < min_days:
                    print(f"  [{start}~{end}] range rejected even at minimum size, giving up on this range")
                    return []
                mid = start + (end - start) // 2
                print(f"  [{start}~{end}] range too long, splitting into {start}~{mid} and "
                      f"{mid + timedelta(days=1)}~{end}")
                return (fetch_range(start, mid, min_days)
                        + fetch_range(mid + timedelta(days=1), end, min_days))

            print(f"  [attempt {attempt}] {start}~{end} status={resp.status_code} body={body}")
        except requests.exceptions.RequestException as e:
            print(f"  [attempt {attempt}] {start}~{end} error={e}")
        if attempt < MAX_RETRIES:
            import time as _t
            _t.sleep(2)
    print(f"  [failed] {start}~{end} gave up after {MAX_RETRIES} attempts")
    return []


def fetch_and_save_mid(start_date: date = START_DATE, end_date: date = END_DATE,
                        output_file: str = OUTPUT_FILE) -> pd.DataFrame:
    chunks = list(daterange_chunks(start_date, end_date, CHUNK_DAYS))
    print(f"Fetching MID data {start_date} to {end_date} in {len(chunks)} initial chunks of "
          f"~{CHUNK_DAYS} days (chunks that Elexon rejects as too long are auto-split smaller)")

    all_rows = []
    printed_sample = False

    for i, (start, end) in enumerate(chunks, start=1):
        print(f"[{i}/{len(chunks)}] {start} to {end}")
        records = fetch_range(start, end)

        if records and not printed_sample:
            print("  Sample record (checking field names):")
            print(" ", records[0])
            printed_sample = True

        for rec in records:
            settlement_date = pick_field(rec, DATE_FIELD_CANDIDATES)
            settlement_period = pick_field(rec, PERIOD_FIELD_CANDIDATES)
            price = pick_field(rec, PRICE_FIELD_CANDIDATES)
            volume = pick_field(rec, VOLUME_FIELD_CANDIDATES)
            start_time = pick_field(rec, START_TIME_FIELD_CANDIDATES)
            provider = pick_field(rec, PROVIDER_FIELD_CANDIDATES)

            if settlement_date is None or settlement_period is None or price is None:
                continue
            all_rows.append({
                "settlementDate": settlement_date,
                "settlementPeriod": settlement_period,
                "startTime": start_time,
                "price": price,
                "volume": volume if volume is not None else 0.0,
                "dataProvider": provider,
            })

    if not all_rows:
        print("\nNo usable records parsed. Check the sample record printed above against "
              "the *_FIELD_CANDIDATES lists at the top of this script, and report back "
              "what the real field names are.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    # Volume-weighted average across data providers (APX, N2EX, ...) for each
    # settlement period, per Elexon's own definition of the overall Market Price:
    #   P_MID = sum(price_j * volume_j) / sum(volume_j)
    def _weighted(group: pd.DataFrame) -> float:
        total_vol = group["volume"].sum()
        if total_vol <= 0:
            return float(group["price"].mean())  # fallback if no volume reported
        return float((group["price"] * group["volume"]).sum() / total_vol)

    grouped = (
        df.groupby(["settlementDate", "settlementPeriod"])
        .apply(lambda g: pd.Series({
            "marketIndexPrice": _weighted(g),
            "startTime": g["startTime"].iloc[0],
        }), include_groups=False)
        .reset_index()
    )
    grouped = grouped.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\nDone: {len(grouped)} settlement-period MID records (volume-weighted across providers)")
    print(f"Saved to: {output_file}")
    return grouped


if __name__ == "__main__":
    fetch_and_save_mid()
