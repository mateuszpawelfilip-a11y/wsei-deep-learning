from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import VECTOR_DIM


class SurrogateMLP(nn.Module):
    def __init__(self, input_dim: int = VECTOR_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SurrogateXGB:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)


def train_surrogates(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    mlp_epochs: int = 200,
) -> tuple[SurrogateMLP, SurrogateXGB, StandardScaler]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = SurrogateXGB()
    xgb.fit(X_train, y_train)

    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32, device=device)
    X_va = torch.tensor(scaler.transform(X_val), dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_va = torch.tensor(y_val, dtype=torch.float32, device=device)

    mlp = SurrogateMLP().to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_state, best_val = None, float("inf")

    for _ in range(mlp_epochs):
        mlp.train()
        opt.zero_grad()
        pred = mlp(X_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        opt.step()
        mlp.eval()
        with torch.no_grad():
            val_loss = loss_fn(mlp(X_va), y_va).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(mlp.state_dict())

    if best_state is not None:
        mlp.load_state_dict(best_state)
    return mlp, xgb, scaler


def predict_surrogates(
    mlp: SurrogateMLP,
    xgb: SurrogateXGB,
    scaler: StandardScaler,
    X: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    Xs = scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
    mlp.eval()
    with torch.no_grad():
        xt = torch.tensor(Xs, dtype=torch.float32, device=device)
        mlp_pred = mlp(xt).cpu().numpy()
    xgb_pred = xgb.predict(Xs)
    if mlp_pred.ndim == 0:
        mlp_pred = np.array([float(mlp_pred)])
    return {
        "mlp": float(mlp_pred[0] if len(mlp_pred) == 1 else mlp_pred.mean()),
        "xgb": float(xgb_pred[0] if len(xgb_pred) == 1 else xgb_pred.mean()),
        "ensemble": float(0.5 * mlp_pred[0] + 0.5 * xgb_pred[0]),
    }
