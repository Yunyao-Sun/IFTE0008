"""
Economic value / VaR analysis v2 -- driven by REAL pointwise prediction data
(from checkpoint re-inference), not Monte Carlo simulation from aggregate RMSE.

Input: {turbine}_{freq}_{experiment}_pointwise.csv files produced by
       inference_extract.py / run_all.py, each with columns:
       group_id, horizon_step, actual_kw, pred_p10, pred_p50, pred_p90
"""

from pathlib import Path

POINTWISE_DIR = Path("./inputs/pointwise")

# Raw Elexon System Price CSV (auto-fetched by main.py via fetch_prices.py
# if missing). GB has used single cash-out pricing since Nov 2015 (BSC
# modification P305), so systemSellPrice == systemBuyPrice in this data --
# this alone cannot be used as a cost reference price, it's only one half
# of the real_mid formula (see build_combined_price_table.py / README).
PRICE_CSV = Path("./inputs/elexon_system_price_2020_2022.csv")

OUTPUT_DIR = Path("./outputs")
SUMMARY_TABLE_CSV = OUTPUT_DIR / "economic_value_summary_v2.csv"

# Settlement mechanism: settlement period is 30 minutes in the GB market
SETTLEMENT_PERIOD_MINUTES = 30

# --- Price mode ---
# "fixed": simplified symmetric £/MWh penalty (default, no external price data needed)
# "real_mid": L_t = (E_t - E_hat_t) * (marketIndexPrice_t - systemPrice_t), using
#             the combined price table built by build_combined_price_table.py.
#             This requires an INDEPENDENT reference price (Market Index Data),
#             NOT SSP/SBP, since GB's single cash-out pricing makes SSP=SBP.
PRICE_MODE = "real_mid"  # "fixed" or "real_mid"

FIXED_ERROR_PENALTY = 15.0   # GBP/MWh, applied to |actual - predicted| energy (PRICE_MODE="fixed")

# Combined real price table (produced by build_combined_price_table.py),
# columns: settlementDate, settlementPeriod, systemPrice, marketIndexPrice, startTime
# only used when PRICE_MODE = "real_mid"
REAL_PRICE_CSV = Path("./inputs/real_price_table_2020_2022.csv")

# --- VaR / CVaR ---
VAR_CONFIDENCE = 0.95
