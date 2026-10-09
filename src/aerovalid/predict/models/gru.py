"""Optional PyTorch GRU residual-prediction head (FR-PRD-006, `[ml]` extra). Guarded
import: if `torch` isn't installed, `HAS_TORCH` is False and constructing `GRUModel`
raises a clear `ImportError` naming the extra to install -- the module itself always
imports cleanly, so the rest of the codebase never fails just because this is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aerovalid.predict.models import numeric_feature_columns

try:
    import torch
    from torch import nn, optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class _GRUHead(nn.Module):  # type: ignore[misc]
        """A length-1-sequence GRU over the (already summary-statistic) feature
        vector, since `predict/datasets.py` produces lag/rolling features rather than
        raw sequences; the GRU still adds a learned nonlinear recurrent transform
        ahead of a linear head."""

        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x.unsqueeze(1))
            return self.head(out[:, -1, :]).squeeze(-1)


class GRUModel:
    """Tiny GRU regressor wrapper; requires the optional `[ml]` extra (`torch`)."""

    def __init__(
        self, hidden_dim: int = 32, epochs: int = 20, lr: float = 1e-3, seed: int = 0
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "GRUModel requires the optional '[ml]' extra (torch); install with "
                "`pip install -e '.[ml]'`."
            )
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self._feature_columns: list[str] = []
        self._net: nn.Module | None = None
        self._feature_mean: NDArray[np.float64] = np.empty(0)
        self._feature_std: NDArray[np.float64] = np.empty(0)
        self._residual_std: float = 1.0

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
        groups: NDArray[np.str_] | None = None,
    ) -> None:
        del groups
        torch.manual_seed(self.seed)
        self._feature_columns = numeric_feature_columns(features)
        x = features[self._feature_columns].to_numpy(dtype=np.float64)
        self._feature_mean = x.mean(axis=0)
        self._feature_std = x.std(axis=0)
        self._feature_std[self._feature_std == 0] = 1.0
        x_norm = (x - self._feature_mean) / self._feature_std

        self._net = _GRUHead(input_dim=x_norm.shape[1], hidden_dim=self.hidden_dim)
        optimiser = optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        x_tensor = torch.tensor(x_norm, dtype=torch.float32)
        y_tensor = torch.tensor(target, dtype=torch.float32)

        self._net.train()
        for _epoch in range(self.epochs):
            optimiser.zero_grad()
            prediction = self._net(x_tensor)
            loss = loss_fn(prediction, y_tensor)
            loss.backward()
            optimiser.step()

        self._net.eval()
        with torch.no_grad():
            residuals = target - self._net(x_tensor).numpy()
        self._residual_std = float(np.std(residuals)) or 1e-9

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        assert self._net is not None, "GRUModel.predict called before fit"
        x = features[self._feature_columns].to_numpy(dtype=np.float64)
        x_norm = (x - self._feature_mean) / self._feature_std
        with torch.no_grad():
            prediction = self._net(torch.tensor(x_norm, dtype=torch.float32)).numpy()
        return np.asarray(prediction, dtype=np.float64)

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.1
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        from scipy.stats import norm as normal_dist

        point = self.predict(features)
        half_width = normal_dist.ppf(1.0 - alpha / 2.0) * self._residual_std
        return point - half_width, point + half_width
