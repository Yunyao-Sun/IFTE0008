# Economic Value / VaR Analysis

This module converts wind-power point forecasts into scenario-level economic
losses and evaluates tail risk using VaR and CVaR.

The analysis is **turbine-level**, not portfolio-level. Forecast errors are
combined with historical Elexon Market Price and System Price data using
historical price scenario revaluation.

## Workflow

The economic model does not train the forecasting models. Forecasting must be
completed first.

### 1. Run the forecasting models

There are 5 turbines and 3 temporal resolutions, giving **15 turbine–resolution
configurations**.

For each configuration, set `TURBINE_ID` and `FREQ` in the forecasting project
and run:

```bash
python main.py
```

Each run automatically loops through all five experiments (E1–E5), so the
forecasting `main.py` is run **15 times in total**, not 75 times.

### 2. Generate pointwise forecasts

For the same 15 turbine–resolution configurations, run:

```bash
python inference.py
```

Each inference run also loops through E1–E5 and extracts test-set pointwise
outputs, including actual power and P50 forecasts. This produces the available
`{turbine}_{freq}_{experiment}_pointwise.csv` files.

`inference.py` only generates pointwise forecasting outputs. It does not
calculate prices, economic losses, VaR, or CVaR.

### 3. Prepare the economic-model input

Copy all available pointwise CSV files into:

```text
inputs/pointwise/
```

### 4. Run the economic model

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

The main analysis uses `PRICE_MODE = "real_mid"` and combines historical
Elexon Market Index Price with System Price. Missing price files are downloaded
and prepared automatically.

## Core calculation

For each settlement period:

```text
loss = (actual_energy - predicted_energy) × (Market Price - System Price)
```

Power forecasts are converted to MWh and aligned to 30-minute settlement
periods before losses are aggregated over each 8-hour forecast scenario.

## Main outputs

Files are written to `outputs/`. The main files used in the dissertation are:

- `economic_value_summary_v2.csv` — model-level VaR/CVaR results
- `incremental_loss_vs_E1.csv` — paired incremental loss and VaR/CVaR relative to E1
- `incremental_loss_vs_E1_detail.csv` — scenario-level paired loss differences
- `raw_costs/` — scenario-level loss files
- `figures/` — generated risk figures

## Important limitations

- Pointwise prediction files do not contain calendar timestamps. Forecast errors
  are therefore paired with continuous historical 8-hour price paths rather than
  the prices from the actual prediction dates. Results are **historical price
  scenario revaluations**, not realised settlement costs.
- Economic losses are calculated separately for each turbine; wind-farm
  portfolio netting is not modelled.
- Price-path assignment currently includes temporal resolution in the random
  seed, so direct economic comparisons across 10-, 20-, and 30-minute
  resolutions should be interpreted cautiously.
