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
        self.lstm    = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq, features)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # apply dropout on last timestep
        return self.fc(out)                # (batch, output_size)

def _make_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def train_bayesian_lstm(
    df_features: pd.DataFrame,
    budget_cols: list,
    lookback=3,
    epochs=50,
    batch_size=16,
    hidden_size=50,
    dropout=0.3,
    lr=1e-3
):
    """
    Trains the BayesianMultiLSTM on df_features (which include
    both budget columns and any exogenous features like inflation).
    Returns trained model and fitted MinMaxScaler.
    """
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(df_features.values)
    X, Y_all = _make_sequences(data_all, lookback)

    # targets: only the budget columns
    idxs = [df_features.columns.get_loc(c) for c in budget_cols]
    Y = Y_all[:, idxs]

    ds  = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32)
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model     = BayesianMultiLSTM(
        input_size=X.shape[2],
        hidden_size=hidden_size,
        output_size=len(budget_cols),
        dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            preds = model(xb)
            loss  = loss_fn(preds, yb)
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
    lookback=3,
    mc_samples=50,
    seed=42
):
    """
    Performs recursive MC-dropout forecasting. Returns two DataFrames:
      - mean_df: mean predictions for each future step
      - std_df : standard deviation (uncertainty) per step
    Uncertainty will naturally grow as the sampled predictions feed into
    subsequent steps.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Prepare scaled sequences
    past_scaled   = scaler.transform(past_df.values).tolist()
    future_scaled = scaler.transform(future_df.values).tolist()
    n_steps       = len(future_scaled)

    model.train()  # keep dropout active
    all_preds = []

    for _ in range(mc_samples):
        buf   = past_scaled.copy()
        preds = []
        for i in range(n_steps):
            seq = torch.tensor([buf[-lookback:]], dtype=torch.float32)
            with torch.no_grad():
                out = model(seq).squeeze(0).numpy()
            preds.append(out)
            # append both predicted budgets + future exogenous features
            buf.append(np.concatenate([out, future_scaled[i][len(budget_cols):]]))
        all_preds.append(preds)

    arr = np.stack(all_preds, axis=0)     # (mc_samples, n_steps, budget_count)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    # Inverse‐scale only budgets
    n_steps, bc = mean.shape
    filler_mean = np.zeros((n_steps, past_df.shape[1]-bc))
    full_mean   = np.hstack([mean, filler_mean])
    inv_mean    = scaler.inverse_transform(full_mean)[:, :bc]

    filler_std  = np.zeros((n_steps, past_df.shape[1]-bc))
    full_std    = np.hstack([std, filler_std])
    inv_std     = scaler.inverse_transform(full_std)[:, :bc]

    years = future_df.index
    mean_df = pd.DataFrame(inv_mean, index=years, columns=budget_cols)
    std_df  = pd.DataFrame(inv_std,  index=years, columns=budget_cols)
    return mean_df, std_df
