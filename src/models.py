from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except Exception:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from config import CFG


class GlobalNormalizer:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, y: np.ndarray) -> "GlobalNormalizer":
        self.min_ = float(np.nanmin(y))
        self.max_ = float(np.nanmax(y))
        print(f"  GlobalNormalizer fit: min={self.min_:.2f}  max={self.max_:.2f}")
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.min_) / (self.max_ - self.min_ + 1e-9)

    def inverse_transform(self, y_norm: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError(
                "GlobalNormalizer.inverse_transform() was called before "
                "fit(). min_/max_ are still None. Call set_global_normalizer() "
                "with an already-fitted scaler (e.g. dataset_builder.TARGET_SCALER) "
                "before running predict_tft() / predict_lstm()."
            )
        return y_norm * (self.max_ - self.min_ + 1e-9) + self.min_

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        self.fit(y)
        return self.transform(y)


GLOBAL_NORM = GlobalNormalizer()


def set_global_normalizer(scaler) -> None:
    global GLOBAL_NORM
    GLOBAL_NORM = scaler


class WindDataset(Dataset):
    def __init__(self, samples: dict):
        self.X = torch.tensor(samples["X"], dtype=torch.float32)
        self.y = torch.tensor(samples["y"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


class QuantileLSTM(pl.LightningModule):
    def __init__(self, input_size: int):
        super().__init__()
        self.save_hyperparameters()
        self.quantiles  = CFG.TFT_QUANTILES
        self.horizon    = CFG.HORIZON
        hidden          = CFG.LSTM_HIDDEN_SIZE

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden,
            num_layers  = CFG.LSTM_LAYERS,
            batch_first = True,
            bidirectional = True,
            dropout     = CFG.LSTM_DROPOUT if CFG.LSTM_LAYERS > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, self.horizon),
            )
            for _ in self.quantiles
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_w = torch.softmax(
            self.attn(lstm_out), dim=1)
        context = (lstm_out * attn_w).sum(dim=1)
        preds = torch.stack(
            [head(context) for head in self.heads], dim=-1
        )
        return preds

    def _quantile_loss(self, pred: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        for i, q in enumerate(self.quantiles):
            err  = target - pred[:, :, i]
            loss = loss + torch.mean(
                torch.maximum(q * err, (q - 1) * err))
        return loss / len(self.quantiles)

    def training_step(self, batch, _):
        x, y = batch
        loss = self._quantile_loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = self._quantile_loss(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt   = torch.optim.Adam(self.parameters(), lr=CFG.LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=3, factor=0.5)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }


def build_tft(training_ds: TimeSeriesDataSet) -> TemporalFusionTransformer:
    model = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate          = CFG.LR,
        hidden_size            = CFG.TFT_HIDDEN_SIZE,
        attention_head_size    = CFG.TFT_ATTN_HEADS,
        dropout                = CFG.TFT_DROPOUT,
        hidden_continuous_size = max(8, CFG.TFT_HIDDEN_SIZE // 2),
        output_size            = len(CFG.TFT_QUANTILES),
        loss                   = QuantileLoss(CFG.TFT_QUANTILES),
        log_interval           = 20,
        reduce_on_plateau_patience = 3,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TFT parameter count: {n_params:,}")
    return model


def _run_tag(exp_name: str) -> str:
    return f"{CFG.TURBINE_ID}_{CFG.FREQ}_{exp_name}"


def train_tft(training_ds: TimeSeriesDataSet,
              val_ds: TimeSeriesDataSet,
              exp_name: str) -> TemporalFusionTransformer:
    ckpt_dir = CFG.CHECKPOINT_DIR / _run_tag(exp_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_tft(training_ds)

    train_loader = training_ds.to_dataloader(
        train=True, batch_size=CFG.BATCH_SIZE, num_workers=0)
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=CFG.BATCH_SIZE * 2, num_workers=0)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=CFG.PATIENCE,
        mode="min", verbose=True)
    checkpoint = ModelCheckpoint(
        monitor="val_loss", dirpath=str(ckpt_dir),
        filename="best", save_top_k=1, mode="min")

    trainer = pl.Trainer(
        max_epochs           = CFG.MAX_EPOCHS,
        accelerator          = "auto",
        devices              = 1,
        callbacks            = [early_stop, checkpoint],
        gradient_clip_val    = 0.1,
        enable_progress_bar  = True,
        enable_model_summary = False,
        log_every_n_steps    = 20,
    )

    print(f"Training TFT for {exp_name}")
    trainer.fit(model, train_loader, val_loader)

    best = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint.best_model_path)
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    return best


def predict_tft(model: TemporalFusionTransformer,
                test_ds: TimeSeriesDataSet,
                inverse: bool = True) -> np.ndarray:
    loader = test_ds.to_dataloader(
        train=False, batch_size=CFG.BATCH_SIZE * 2, num_workers=0)
    preds = model.predict(loader, mode="quantiles")
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    else:
        preds = np.asarray(preds)

    if inverse:
        preds = GLOBAL_NORM.inverse_transform(preds)
    return preds


def train_lstm(train_samples: dict, val_samples: dict,
               exp_name: str) -> QuantileLSTM:
    ckpt_dir = CFG.CHECKPOINT_DIR / _run_tag(exp_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    input_size = train_samples["X"].shape[2]
    model      = QuantileLSTM(input_size)
    n_params   = sum(p.numel() for p in model.parameters())
    print(f"LSTM parameter count: {n_params:,}")

    train_loader = DataLoader(
        WindDataset(train_samples),
        batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        WindDataset(val_samples),
        batch_size=CFG.BATCH_SIZE * 2, num_workers=0)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=CFG.PATIENCE, mode="min")
    checkpoint = ModelCheckpoint(
        monitor="val_loss", dirpath=str(ckpt_dir),
        filename="best", save_top_k=1, mode="min")

    trainer = pl.Trainer(
        max_epochs           = CFG.MAX_EPOCHS,
        accelerator          = "auto",
        devices              = 1,
        callbacks            = [early_stop, checkpoint],
        enable_progress_bar  = True,
        enable_model_summary = False,
        log_every_n_steps    = 20,
    )

    print(f"Training LSTM for {exp_name}")
    trainer.fit(model, train_loader, val_loader)

    best = QuantileLSTM.load_from_checkpoint(
        checkpoint.best_model_path, input_size=input_size)
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    return best


def predict_lstm(model: QuantileLSTM,
                 test_samples: dict,
                 inverse: bool = True) -> np.ndarray:
    model.eval()
    X = torch.tensor(test_samples["X"], dtype=torch.float32)
    with torch.no_grad():
        preds = model(X).detach().cpu().numpy()

    if inverse:
        preds = GLOBAL_NORM.inverse_transform(preds)
    return preds
