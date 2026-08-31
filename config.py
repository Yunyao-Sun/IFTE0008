import re
from pathlib import Path


def _freq_to_minutes(freq: str) -> int:
    m = re.match(r"\s*(\d+)\s*min", str(freq).strip().lower())
    if not m:
        raise ValueError(f"Unsupported FREQ format: {freq!r}. Expected e.g. '10min'.")
    return int(m.group(1))


class Config:
    ROOT           = Path(__file__).resolve().parent
    DATA_DIR       = ROOT / "data"
    SRC_DIR        = ROOT / "src"
    RESULTS_DIR    = ROOT / "results"
    CHECKPOINT_DIR = ROOT / "checkpoints"

    DATA_PATTERNS  = ["*.xlsx", "*.xls", "*.csv"]
    REQUIRED_TARGET_NAMES = [
        "Power (kW)", "Active power (kW)", "Active Power (kW)",
        "Power", "Active power", "Grid active power",
        "Grid active power (kW)", "Electrical power (kW)",
    ]

    TURBINE_ID   = "WT9"

    TARGET_COL   = "power_kw"
    RATED_POWER  = 2050.0
    CUT_IN_SPEED = 3.5
    CUT_OUT_SPEED = 25.0

    FREQ = "30min"

    LOOKBACK_HOURS = 24
    HORIZON_HOURS  = 8
    STEP_HOURS     = 5

    @property
    def FREQ_MINUTES(self) -> int:
        return _freq_to_minutes(self.FREQ)

    @property
    def WINDOW_SIZE(self) -> int:
        return round(self.LOOKBACK_HOURS * 60 / self.FREQ_MINUTES)

    @property
    def HORIZON(self) -> int:
        return round(self.HORIZON_HOURS * 60 / self.FREQ_MINUTES)

    @property
    def STEP(self) -> int:
        return round(self.STEP_HOURS * 60 / self.FREQ_MINUTES)

    MAX_SAMPLES_PER_SPLIT = None

    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15

    TIME_COLS = [
        "hour_sin", "hour_cos",
        "dow_sin",  "dow_cos",
        "month_sin","month_cos",
    ]

    MAX_MISSING_RATIO = 0.30
    SHORT_GAP_LIMIT   = 3

    VMD_K        = 5
    VMD_ALPHA    = 2000.0
    VMD_TAU      = 0.0
    VMD_TOL      = 1e-6
    VMD_MAX_ITER = 250

    EN_TOP_N   = 12
    EN_ALPHA   = 0.001
    EN_L1_RATIO = 0.70

    TFT_NORMALIZER  = "global_minmax"

    TFT_HIDDEN_SIZE = 128
    TFT_ATTN_HEADS  = 4
    TFT_DROPOUT     = 0.10
    TFT_QUANTILES   = [0.1, 0.5, 0.9]

    LSTM_HIDDEN_SIZE = 128
    LSTM_LAYERS      = 2
    LSTM_DROPOUT     = 0.10

    BATCH_SIZE  = 64
    MAX_EPOCHS  = 50
    LR          = 1e-3
    PATIENCE    = 8

    MAPE_THRESHOLD = 0.05

    SEED = 42

    EXP_NAMES = {
        1: "E1_raw_tft",
        2: "E2_global_mvmd_tft",
        3: "E3_rolling_mvmd_tft",
        4: "E4_rolling_mvmd_en_lstm",
        5: "E5_rolling_mvmd_en_tft",
    }

    def __init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)


CFG = Config()
