# scripts/train_lstm.py

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
VOL_COL = "brent_vol_20"


# ---------------- Attention ---------------- #

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        score = self.v(torch.tanh(self.W(h))).squeeze(-1)
        w = torch.softmax(score, dim=1)
        return torch.sum(h * w.unsqueeze(-1), dim=1)


# ---------------- Model ---------------- #

class LSTMAttentionModel(LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, lr):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attn = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0) # Huber loss

    def forward(self, x):
        h, (hn, _) = self.lstm(x)
        ctx = self.attn(h)
        last = hn[-1]
        out = self.fc(torch.cat([ctx, last], dim=1))
        return torch.tanh(out).squeeze(-1)  # implicit clamp

    def training_step(self, batch, _):
        X, y = batch
        loss = self.loss_fn(self(X), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        X, y = batch
        loss = self.loss_fn(self(X), y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ---------------- Data ---------------- #

def _extract_xy(df):
    logret = df["log_ret"]
    vol = df[VOL_COL].replace(0, np.nan)

    y = (logret * 100 / vol).fillna(0.0)
    X = df.drop(columns=["log_ret"])

    return X.astype(float), y.astype(float), list(X.columns)


# ---------------- Train ---------------- #

def train_and_get_model(train_df, params):
    seq_len = params["seq_len"]
    X, y, cols = _extract_xy(train_df)

    n = int(len(X) * 0.9)
    scaler = StandardScaler().fit(X.iloc[:n])

    Xs = pd.DataFrame(scaler.transform(X), index=X.index, columns=cols)

    tr = SeqDataset(Xs.iloc[:n], y.iloc[:n], seq_len)
    va = SeqDataset(Xs.iloc[n-seq_len:], y.iloc[n-seq_len:], seq_len)

    model = LSTMAttentionModel(
        input_dim=len(cols),
        hidden_dim=params["hidden_dim"],
        num_layers=params["layers"],
        dropout=params["dropout"],
        lr=params["lr"]
    )

    trainer = Trainer(
        max_epochs=params["epochs"],
        accelerator="auto",
        logger=False,
        callbacks=[
            EarlyStopping("val_loss", patience=params["early_stopping"]),
            ModelCheckpoint(params.get("save_path","models/lstm"), monitor="val_loss")
        ]
    )

    trainer.fit(
        model,
        DataLoader(tr, batch_size=params["batch_size"], shuffle=True),
        DataLoader(va, batch_size=params["batch_size"])
    )

    best = LSTMAttentionModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    return {
        "model": best.eval(),
        "scaler": scaler,
        "feature_cols": cols,
        "seq_len": seq_len,
        "close_col": CLOSE_COL,
        "vol_col": VOL_COL
    }


# ---------------- Predict ---------------- #

@torch.no_grad()
def predict_recursive(bundle, full_df, start_price, dates, bias_correction=0.0):
    """
    Runs recursive prediction with optional Static Anchor (Bias Correction).
    
    Args:
        bias_correction (float): The offset to add to the final output. 
                                 Calculated as (Actual_Today - Model_Belief_Today).
                                 This lifts the curve to match reality without breaking recursion.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = bundle["model"].to(device)
    scaler = bundle["scaler"]
    cols = bundle["feature_cols"]
    seq_len = bundle["seq_len"]
    close_col = bundle["close_col"]
    vol_col = bundle["vol_col"]
    
    df = full_df.sort_index()
    anchor_vol = float(df[vol_col].iloc[-1])
    # scale full history ONCE (allowed)
    Xs = pd.DataFrame(
        scaler.transform(df[cols]),
        index=df.index,
        columns=cols
    )

    # initial window (last known history before prediction)
    #always anchor on last available date
    pos = len(Xs) - 1

    window = torch.tensor(
        Xs.iloc[pos - seq_len + 1 : pos + 1].values,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)  # (1, T, F)

    close_idx = cols.index(close_col)
    price = float(start_price)
    preds = []

    model.eval()

    for d in dates:
        # predict normalized target
        z = float(model(window).detach().cpu())

        # exogenous volatility (allowed)
        vol = anchor_vol
        logret = (z * vol) / 100.0

        # update price (Internal Recursive State)
        price *= np.exp(logret)

        # ------------------- BIAS CORRECTION -------------------
        # Apply the Static Anchor offset to the OUTPUT only.
        # The internal 'price' variable remains untouched to preserve recursive logic.
        final_output_price = price + bias_correction
        # -------------------------------------------------------

        preds.append(final_output_price)

        # ---- CAUSAL WINDOW UPDATE ----
        # take last feature vector from window itself
        last_scaled = window[0, -1].detach().cpu().numpy().reshape(1, -1)
        last_unscaled = scaler.inverse_transform(last_scaled)[0]

        # update ONLY price-derived feature
        # Note: We use the internal 'price' (without bias) to keep the model consistent
        last_unscaled[close_idx] = price

        # rescale
        new_scaled = scaler.transform(last_unscaled.reshape(1, -1))[0]

        new_row = torch.tensor(
            new_scaled,
            dtype=torch.float32,
            device=device
        ).view(1, 1, -1)

        window = torch.cat([window[:, 1:], new_row], dim=1)

    return pd.Series(preds, index=dates, name="lstm_pred")