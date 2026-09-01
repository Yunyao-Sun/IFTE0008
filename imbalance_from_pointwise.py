"""
Core logic: turn real pointwise (actual vs predicted power) data into
per-forecast-scenario turbine-level imbalance cost.

Two price modes (config.PRICE_MODE):

  "fixed" (default): cost = FIXED_ERROR_PENALTY * |imbalance_energy| / 1000
      No external price data needed.

  "real_mid": cost = imbalance_energy * (MID - SystemPrice) / 1000
      Uses two genuinely independent real price series (Market Index Data
      and GB System Price). Signed, not absolute value.

Deterministic per-scenario price paths
----------------------------------------
Each 8-hour forecast scenario in "real_mid" mode is assigned a historical
price path drawn from build_scenario_seed(turbine, freq, group_id) -- a
hash of the scenario's own identity, NOT a running RNG state. This
guarantees the SAME (turbine, freq, group_id) always gets the SAME price
path regardless of which experiment is being processed, what order
experiments run in, or whether some other experiment is missing a group_id.
This is required for paired incremental-loss analysis (E_m vs E1 for the
same scenario) to be valid -- price cannot be a hidden confound.

Continuity check
------------------
A price path is only valid if the underlying settlement periods are truly
consecutive (30 minutes apart based on startTime), not just consecutive
rows in a possibly-gappy table (inner joins / missing days can create
silent gaps). Valid starting positions for a given path length are
precomputed once per price table.

Settlement period alignment
----------------------------
Forecast steps (10/20/30 min) are redistributed into true 30-minute GB
settlement periods, weighted by minutes of overlap -- this applies
identically in both price modes and handles the 20-minute case (30/20=1.5,
does not divide evenly) correctly.
"""

import hashlib

import numpy as np
import pandas as pd

import config

SETTLEMENT_MINUTES = config.SETTLEMENT_PERIOD_MINUTES
SCENARIO_HORIZON_PERIODS = 16  # 8 hours / 30 min, true for every granularity


def redistribute_to_settlement_periods(power_kw: np.ndarray, step_minutes: int,
                                        settlement_minutes: int = SETTLEMENT_MINUTES) -> np.ndarray:
    """
    Convert per-step average power (kW) at native forecast granularity into
    energy (kWh) per 30-min settlement period, allocating each step's
    energy proportionally to overlap minutes with each settlement period.
    """
    n_steps = len(power_kw)
    total_minutes = n_steps * step_minutes
    n_periods = int(round(total_minutes / settlement_minutes))

    energies = np.zeros(n_periods)
    for step_idx in range(n_steps):
        s_start = step_idx * step_minutes
        s_end = s_start + step_minutes
        power = power_kw[step_idx]
        if power == 0:
            continue
        first_period = s_start // settlement_minutes
        last_period = (s_end - 1) // settlement_minutes
        for period_idx in range(int(first_period), int(last_period) + 1):
            p_start = period_idx * settlement_minutes
            p_end = p_start + settlement_minutes
            overlap = max(0, min(s_end, p_end) - max(s_start, p_start))
            if overlap > 0 and period_idx < n_periods:
                energies[period_idx] += power * (overlap / 60.0)
    return energies


def build_scenario_seed(base_seed: int, turbine: str, freq: str, group_id) -> int:
    """
    Deterministic seed derived purely from scenario identity, not from
    call order. Same (turbine, freq, group_id) -> same seed, always,
    regardless of which experiment or in what sequence it's processed.
    """
    key = f"{base_seed}|{turbine}|{freq}|{group_id}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:8], 16)


def find_valid_start_indices(price_df: pd.DataFrame, n_periods: int) -> np.ndarray:
    """
    A starting index i is valid only if rows i..i+n_periods-1 are
    genuinely chronologically contiguous (exactly 30 min apart each step,
    based on startTime) -- protects against silent gaps from missing
    settlement periods after the System Price / MID inner join.
    """
    start_times = pd.to_datetime(price_df["startTime"], utc=True).to_numpy()
    diffs_minutes = np.diff(start_times).astype("timedelta64[m]").astype(float)
    is_break = diffs_minutes != SETTLEMENT_MINUTES  # True where a gap/break occurs

    n = len(price_df)
    valid = []
    # run-length: for each position, how many consecutive good steps follow
    good_run = np.zeros(n, dtype=int)
    good_run[-1] = 1
    for i in range(n - 2, -1, -1):
        good_run[i] = 1 if is_break[i] else good_run[i + 1] + 1

    for i in range(n - n_periods + 1):
        if good_run[i] >= n_periods:
            valid.append(i)
    return np.array(valid, dtype=int)


def sample_real_mid_price_path(seed: int, n_periods: int, price_df: pd.DataFrame,
                                valid_start_indices: np.ndarray):
    """
    Deterministically pick one truly-contiguous n_periods-long historical
    price path, using `seed` (derived from scenario identity, see
    build_scenario_seed) to choose WHICH valid contiguous window to use.
    Returns list of (marketIndexPrice, systemPrice) tuples, length n_periods.
    """
    if len(valid_start_indices) == 0:
        raise ValueError(
            f"No contiguous {n_periods}-period (8-hour) window exists in the "
            f"price table without a gap. Check price data coverage."
        )
    rng = np.random.default_rng(seed)
    start_idx = int(valid_start_indices[rng.integers(0, len(valid_start_indices))])
    block = price_df.iloc[start_idx:start_idx + n_periods]
    return list(zip(block["marketIndexPrice"].to_numpy(), block["systemPrice"].to_numpy()))


def compute_sample_costs(
    pointwise_df: pd.DataFrame,
    step_minutes: int,
    turbine: str,
    freq: str,
    price_df: pd.DataFrame = None,
    valid_start_indices: np.ndarray = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each test sample (group_id), compute the TOTAL forecast-error
    economic cost for its 8-hour scenario, using REAL actual-vs-predicted
    values. No distribution assumption, no Monte Carlo sampling of the
    imbalance itself.

    Returns a DataFrame with columns [group_id, scenario_loss, mean_spread,
    min_spread, max_spread] -- group_id is preserved so scenarios can later
    be paired across experiments (e.g. E5's group_37 vs E1's group_37) for
    incremental-loss analysis. mean/min/max_spread record the (MID -
    SystemPrice) values actually drawn for this scenario's price path
    (NaN in "fixed" mode, where no external price is used) -- useful for
    a descriptive scatter of loss vs sampled spread magnitude, but note
    this does NOT establish that the forecast error and this price spread
    co-occurred in reality (see README: no date-matching between forecast
    error and price).
    """
    expected_minutes = 8 * 60
    rows = []

    for group_id, group_df in pointwise_df.groupby("group_id"):
        group_df = group_df.sort_values("horizon_step")

        actual_minutes = len(group_df) * step_minutes
        if actual_minutes != expected_minutes:
            raise ValueError(
                f"group_id={group_id}: expected {expected_minutes} minutes of "
                f"forecast horizon, got {len(group_df)} steps = {actual_minutes} "
                f"minutes. This pointwise file may be truncated or corrupted."
            )
        steps = group_df["horizon_step"].to_numpy()
        if not np.array_equal(steps, np.arange(len(steps))):
            raise ValueError(
                f"group_id={group_id}: horizon_step values are not contiguous "
                f"0..{len(steps)-1} (got {steps.tolist()}). File may be missing rows."
            )

        actual = group_df["actual_kw"].to_numpy()
        pred = group_df["pred_p50"].to_numpy()

        actual_energy = redistribute_to_settlement_periods(actual, step_minutes)
        pred_energy = redistribute_to_settlement_periods(pred, step_minutes)
        imbalance = actual_energy - pred_energy

        if config.PRICE_MODE == "fixed":
            total_loss = float(np.sum(config.FIXED_ERROR_PENALTY * np.abs(imbalance) / 1000.0))
            mean_spread = min_spread = max_spread = float("nan")

        elif config.PRICE_MODE == "real_mid":
            scenario_seed = build_scenario_seed(seed, turbine, freq, group_id)
            price_path = sample_real_mid_price_path(
                scenario_seed, len(imbalance), price_df, valid_start_indices)
            spreads = np.array([mid - sysp for mid, sysp in price_path])
            mean_spread, min_spread, max_spread = spreads.mean(), spreads.min(), spreads.max()
            total_loss = float(sum(
                imb * spread / 1000.0
                for imb, spread in zip(imbalance, spreads)
            ))
        else:
            raise ValueError(f"Unknown config.PRICE_MODE: {config.PRICE_MODE!r}")

        rows.append({
            "group_id": group_id, "scenario_loss": total_loss,
            "mean_spread": mean_spread, "min_spread": min_spread, "max_spread": max_spread,
        })

    return pd.DataFrame(rows)
