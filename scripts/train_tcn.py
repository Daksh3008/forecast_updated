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


# ---------------------------------------------------------
# Local Multi-Head Attention
# ---------------------------------------------------------

class LocalAttention(nn.Module):
    def __init__(self, channels, heads=4, window=8):
        super().__init__()
        self.heads = heads
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
            x_local = x[:, -self.window:, :]
        else:
            x_local = x

        attn_out, _ = self.attn(x_local, x_local, x_local)
        out = attn_out[:, -1, :]  # (B, C)
        return out


# ---------------------------------------------------------
# TCN Block
# ---------------------------------------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp]


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        padding = (k - 1) * d

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, dilation=d, padding=padding),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, k, dilation=d, padding=padding),
            Chomp1d(padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# TCN + Local Attention (Reactive Model)
# ---------------------------------------------------------

class TCNReactive(LightningModule):
    def __init__(self,
                 input_dim,
                 channels=48,
                 layers=4,
                 lr=1e-3,
                 attn_heads=4,
                 attn_window=8,
                 dropout=0.1):
        super().__init__()
        self.save_hyperparameters()

        blocks = []
        in_ch = input_dim
        for i in range(layers):
            dil = 2 ** i
            blocks.append(TCNBlock(in_ch, channels, k=3, d=dil))
            in_ch = channels

        self.tcn = nn.Sequential(*blocks)
        self.attn = LocalAttention(channels, heads=attn_heads, window=attn_window)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1)
        )

        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)     # (B, F, T)
        h = self.tcn(x)           # (B, C, T)
        h_attn = self.attn(h)     # (B, C)
        out = self.fc(h_attn)     # (B, 1)
        return out.squeeze(-1)

    def training_step(self, batch, _):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ---------------------------------------------------------
# TRAINING WRAPPER
# ---------------------------------------------------------

def _extract_xy(train_df: pd.DataFrame):
    if "log_ret" not in train_df.columns:
        raise ValueError("Expected 'log_ret' in train_df")

    y = train_df["log_ret"].astype(float)
    feature_cols = [c for c in train_df.columns if c != "log_ret"]

    X = train_df[feature_cols].astype(float)
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

    n_total = len(train_df)
    n_train = int(n_total * 0.9)

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]

    X_val = X.iloc[n_train - seq_len:]
    y_val = y.iloc[n_train - seq_len:]

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_s = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=feature_cols)
    X_val_s = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=feature_cols)

    train_ds = SeqDataset(X_train_s, y_train, seq_len=seq_len)
    val_ds = SeqDataset(X_val_s, y_val, seq_len=seq_len)

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
        callbacks=[ckpt, es],
    )

    trainer.fit(model, train_loader, val_loader)
    best_model = TCNReactive.load_from_checkpoint(ckpt.best_model_path)

    return {
        "model": best_model.eval(),
        "scaler": scaler,
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "close_col": CLOSE_COL
    }


@torch.no_grad()
def predict_recursive(model_bundle, full_df, start_price=None, predict_dates=None):
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_cols = model_bundle["feature_cols"]
    seq_len = model_bundle["seq_len"]
    close_col = model_bundle["close_col"]

    df = full_df.sort_index()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ---- Scale all features once ----
    X_all = scaler.transform(df[feature_cols])
    X_all = pd.DataFrame(X_all, index=df.index, columns=feature_cols)

    # ---- Find start window ----
    first_pred = predict_dates[0]
    hist_idx = X_all.index[X_all.index < first_pred]

    if len(hist_idx) < seq_len:
        raise ValueError("Not enough history before first predict date.")

    start_date = hist_idx[-1]
    pos = X_all.index.get_loc(start_date)
    window_idx = X_all.index[pos - seq_len + 1 : pos + 1]

    window = torch.tensor(
        X_all.loc[window_idx].values,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)   # (1, seq_len, F)

    close_idx = feature_cols.index(close_col)
    curr_price = float(start_price)

    mu_list = []

    model.eval()

    # ---- Recursive loop ----
    for _ in range(len(predict_dates)):
        # forward pass → only mu
        mu = model(window)
        mu_val = float(mu.squeeze().detach().cpu().numpy())
        mu_list.append(mu_val)

        # update price
        curr_price *= np.exp(mu_val)

        # rebuild next row
        last_scaled = window[0, -1].detach().cpu().numpy().reshape(1, -1)
        last_unscaled = scaler.inverse_transform(last_scaled)[0]

        # update close
        last_unscaled[close_idx] = curr_price

        # rescale
        new_scaled = scaler.transform(last_unscaled.reshape(1, -1))[0]

        new_row = torch.tensor(
            new_scaled,
            dtype=torch.float32,
            device=device
        ).view(1, 1, -1)

        # slide window
        window = torch.cat([window[:, 1:, :], new_row], dim=1)

    # ---- return only mu series ----
    return pd.Series(mu_list, index=predict_dates, name="tcn_mean_logret")
