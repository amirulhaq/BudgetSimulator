# models/lstm_model.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class BayesianMultiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)               # (batch, seq_len, hidden)
        out = self.dropout(out[:, -1, :])   # apply dropout on last timestep
        return self.fc(out)                 # (batch, output_size)

def _make_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def train_bayesian_lstm(
    df_features: pd.DataFrame,
    budget_cols: list,
    lookback: int = 3,
    epochs: int = 50,
    batch_size: int = 16,
    hidden_size: int = 50,
    lr: float = 1e-3,
    dropout: float = 0.3,
):
    """
    Trains the BayesianMultiLSTM and returns (model, scaler).
    """
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(df_features.values)      # (T, F)
    X, Y_all = _make_sequences(data_all, lookback)           # X:(N,lb,F), Y_all:(N,F)

    # Only budgets are targets
    idxs = [df_features.columns.get_loc(c) for c in budget_cols]
    Y = Y_all[:, idxs]                                        # (N, len(budget_cols))

    # DataLoader
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X_t, Y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = BayesianMultiLSTM(
        input_size=X.shape[2],
        hidden_size=hidden_size,
        output_size=len(budget_cols),
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, scaler

def predict_with_uncertainty(
    past_df: pd.DataFrame,
    future_df: pd.DataFrame,
    model: nn.Module,
    scaler: MinMaxScaler,
    budget_cols: list,
    lookback: int = 3,
    mc_samples: int = 50
):
    """
    Returns (mean_df, std_df) for the next n future steps,
    where n = len(future_df).
    """
    # Scale historical and future features
    past_scaled   = scaler.transform(past_df.values).tolist()
    future_scaled = scaler.transform(future_df.values).tolist()
    n_steps = future_df.shape[0]

    model.train()  # keep dropout ON
    all_preds = []

    for _ in range(mc_samples):
        buf = past_scaled.copy()
        preds = []
        for i in range(n_steps):
            seq = torch.tensor([buf[-lookback:]], dtype=torch.float32)
            with torch.no_grad():
                out = model(seq).squeeze(0).numpy()    # (budget_count,)
            preds.append(out)
            # append budgets + remaining features for this future step
            buf.append(list(out) + future_scaled[i][len(budget_cols):])
        all_preds.append(preds)

        arr = np.array(all_preds)      # (mc_samples, n_steps, budget_count)
    mean = arr.mean(axis=0)        # still in [0,1]
    std  = arr.std(axis=0)

    # === inverse‐scale the budgets ===
    n_steps, bc = mean.shape
    # filler for the non‐budget features
    filler_mean = np.zeros((n_steps, past_df.shape[1] - bc))
    full_mean   = np.hstack([mean, filler_mean])
    inv_mean    = scaler.inverse_transform(full_mean)
    mean_budgets = inv_mean[:, :bc]

    filler_std  = np.zeros((n_steps, past_df.shape[1] - bc))
    full_std    = np.hstack([std, filler_std])
    inv_std     = scaler.inverse_transform(full_std)
    std_budgets  = inv_std[:, :bc]

    years = future_df.index.tolist()
    mean_df = pd.DataFrame(mean_budgets, index=years, columns=budget_cols)
    std_df  = pd.DataFrame(std_budgets,  index=years, columns=budget_cols)
    return mean_df, std_df

