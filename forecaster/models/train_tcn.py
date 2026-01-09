# forecaster/models/train_tcn.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

from scripts.dataset_builder import SeqDataset

CLOSE_COL = "brent_Close"


# ==========================================================
# Local Multi-Head Attention
# ==========================================================

class LocalAttention(nn.Module):
    def __init__(self, channels, heads=4, window=8):
        super().__init__()
        self.window = window
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True
        )

    def forward(self, x):
        # x: (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)

        if x.size(1) > self.window:
            x = x[:, -self.window:, :]

        out, _ = self.attn(x, x, x)
        return out[:, -1, :]   # (B, C)


# ==========================================================
# TCN Block
# ==========================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp]


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        pad = (k - 1) * d

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, dilation=d, padding=pad),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, k, dilation=d, padding=pad),
            Chomp1d(pad),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# ==========================================================
# TCN + Local Attention
# ==========================================================

class TCNReactive(LightningModule):
    def __init__(
        self,
        input_dim,
        channels=48,
        layers=4,
        lr=1e-3,
        attn_heads=4,
        attn_window=8,
        dropout=0.1
    ):
        super().__init__()
        self.save_hyperparameters()

        blocks = []
        in_ch = input_dim
        for i in range(layers):
            d = 2 ** i
            blocks.append(TCNBlock(in_ch, channels, k=3, d=d))
            in_ch = channels

        self.tcn = nn.Sequential(*blocks)
        self.attn = LocalAttention(channels, attn_heads, attn_window)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)     # (B, F, T)
        h = self.tcn(x)           # (B, C, T)
        h = self.attn(h)          # (B, C)
        out = self.fc(h)          # (B, 1)
        return out.squeeze(-1)

    def training_step(self, batch, _):
        X, y = batch
        loss = self.loss_fn(self(X), y)
        return loss

    def validation_step(self, batch, _):
        X, y = batch
        loss = self.loss_fn(self(X), y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ==========================================================
# Training Wrapper (UNCHANGED LOGIC)
# ==========================================================

def _extract_xy(df: pd.DataFrame):
    if "log_ret" not in df.columns:
        raise ValueError("Expected 'log_ret' in dataframe")

    y = df["log_ret"].astype(float)
    feature_cols = [c for c in df.columns if c != "log_ret"]
    X = df[feature_cols].astype(float)
    return X, y, feature_cols


def train_and_get_model(train_df: pd.DataFrame, params: dict):
    seq_len = params.get("seq_len", 60)
    batch_size = params.get("batch_size", 64)
    epochs = params.get("epochs", 40)
    channels = params.get("channels", 48)
    layers = params.get("layers", 4)
    lr = params.get("lr", 1e-3)
    attn_heads = params.get("attn_heads", 4)
    attn_window = params.get("attn_window", 8)
    dropout = params.get("dropout", 0.1)

    save_path = params.get("save_path", "models/tcn")
    os.makedirs(save_path, exist_ok=True)

    X, y, feature_cols = _extract_xy(train_df)

    n = int(len(train_df) * 0.9)

    X_tr = X.iloc[:n]
    y_tr = y.iloc[:n]

    X_va = X.iloc[n - seq_len:]
    y_va = y.iloc[n - seq_len:]

    scaler = StandardScaler().fit(X_tr)

    X_tr_s = pd.DataFrame(scaler.transform(X_tr), index=X_tr.index, columns=feature_cols)
    X_va_s = pd.DataFrame(scaler.transform(X_va), index=X_va.index, columns=feature_cols)

    train_ds = SeqDataset(X_tr_s, y_tr, seq_len)
    val_ds = SeqDataset(X_va_s, y_va, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TCNReactive(
        input_dim=len(feature_cols),
        channels=channels,
        layers=layers,
        lr=lr,
        attn_heads=attn_heads,
        attn_window=attn_window,
        dropout=dropout
    )

    ckpt = ModelCheckpoint(
        dirpath=save_path,
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    es = EarlyStopping(monitor="val_loss", patience=6, mode="min")

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        logger=False,
        callbacks=[ckpt, es]
    )

    trainer.fit(model, train_loader, val_loader)
    best = TCNReactive.load_from_checkpoint(ckpt.best_model_path)

    return {
        "model": best.eval(),
        "scaler": scaler,
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "close_col": CLOSE_COL
    }


# ==========================================================
# FORECAST-SAFE RECURSIVE PREDICTION
# ==========================================================

@torch.no_grad()
def predict_recursive(bundle, full_df, start_price, predict_dates):
    """
    Forecast-safe recursive TCN prediction.

    - Anchor = last historical row
    - No future date index lookups
    - predict_dates used ONLY as labels
    """

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    seq_len = bundle["seq_len"]
    close_col = bundle["close_col"]

    df = full_df.sort_index()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Scale all historical features once
    X_all = pd.DataFrame(
        scaler.transform(df[feature_cols]),
        index=df.index,
        columns=feature_cols
    )

    # ---- FORECAST-SAFE ANCHOR ----
    pos = len(X_all) - 1
    window = torch.tensor(
        X_all.iloc[pos - seq_len + 1: pos + 1].values,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)  # (1, T, F)

    close_idx = feature_cols.index(close_col)
    curr_price = float(start_price)

    mu_list = []

    for _ in predict_dates:
        mu = float(model(window).detach().cpu().numpy())
        mu_list.append(mu)

        curr_price *= np.exp(mu)

        # rebuild next input row
        last_scaled = window[0, -1].detach().cpu().numpy().reshape(1, -1)
        last_unscaled = scaler.inverse_transform(last_scaled)[0]
        last_unscaled[close_idx] = curr_price
        new_scaled = scaler.transform(last_unscaled.reshape(1, -1))[0]

        new_row = torch.tensor(
            new_scaled,
            dtype=torch.float32,
            device=device
        ).view(1, 1, -1)

        window = torch.cat([window[:, 1:], new_row], dim=1)

    return pd.Series(mu_list, index=predict_dates, name="tcn_mean_logret")
