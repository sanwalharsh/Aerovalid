"""Residual normalisation (FR-DET-001): a robust median/MAD baseline fit on nominal
(fault-free) flights, optionally per condition bin, used to whiten every detector's
input into a dimensionless z-score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_MAD_TO_SIGMA = 1.4826  # consistent-with-normal scale factor for the median absolute deviation
_GLOBAL_BIN = "_global_"


@dataclass
class ResidualBaseline:
    """`center`/`scale` per signal, keyed by condition-bin value (`"_global_"` is the
    always-present fallback used when a bin has too few nominal samples)."""

    signals: list[str]
    bin_column: str | None
    center: dict[str, dict[str, float]] = field(default_factory=dict)
    scale: dict[str, dict[str, float]] = field(default_factory=dict)

    def z_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """`z_<signal> = (res_<signal> - center) / scale`, one row per `df` row, using
        the per-bin baseline where `self.bin_column` has enough nominal data and
        falling back to the global baseline otherwise."""
        bins = (
            df[self.bin_column].astype(str).to_numpy()
            if self.bin_column is not None and self.bin_column in df.columns
            else np.full(len(df), _GLOBAL_BIN)
        )
        out: dict[str, np.ndarray] = {}
        for signal in self.signals:
            res_col = f"res_{signal}"
            if res_col not in df.columns:
                continue
            residual = df[res_col].to_numpy(dtype=np.float64)
            center = np.array(
                [self.center.get(str(b), self.center[_GLOBAL_BIN])[signal] for b in bins]
            )
            scale = np.array(
                [self.scale.get(str(b), self.scale[_GLOBAL_BIN])[signal] for b in bins]
            )
            out[f"z_{signal}"] = (residual - center) / scale
        return pd.DataFrame(out)

    def to_json(self) -> str:
        return json.dumps(
            {
                "signals": self.signals,
                "bin_column": self.bin_column,
                "center": self.center,
                "scale": self.scale,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> ResidualBaseline:
        data = json.loads(text)
        return cls(
            signals=data["signals"],
            bin_column=data["bin_column"],
            center=data["center"],
            scale=data["scale"],
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> ResidualBaseline:
        return cls.from_json(Path(path).read_text())


def _median_mad(residual: np.ndarray) -> tuple[float, float]:
    valid = residual[np.isfinite(residual)]
    if valid.size == 0:
        return 0.0, 1.0
    center = float(np.median(valid))
    scale = float(np.median(np.abs(valid - center))) * _MAD_TO_SIGMA
    return center, scale or 1e-9


def fit_baseline(
    nominal_df: pd.DataFrame,
    signals: list[str],
    bin_column: str | None = None,
    min_bin_samples: int = 30,
) -> ResidualBaseline:
    """
    Fit a robust median/MAD residual baseline on nominal (fault-free) flights
    (FR-DET-001).

    Parameters
    ----------
    nominal_df : pandas.DataFrame
        Concatenated aligned rows from fault-free flights only.
    signals : list[str]
        Signals to normalise (uses each `res_<signal>` column).
    bin_column : str, optional
        A `cond_*` column to fit a separate baseline per bin value; bins with fewer
        than `min_bin_samples` rows fall back to the global baseline at lookup time.
    min_bin_samples : int, default 30

    Returns
    -------
    ResidualBaseline
    """
    baseline = ResidualBaseline(signals=list(signals), bin_column=bin_column)
    global_center: dict[str, float] = {}
    global_scale: dict[str, float] = {}
    for signal in signals:
        res_col = f"res_{signal}"
        if res_col not in nominal_df.columns:
            continue
        center, scale = _median_mad(nominal_df[res_col].to_numpy(dtype=np.float64))
        global_center[signal] = center
        global_scale[signal] = scale
    baseline.center[_GLOBAL_BIN] = global_center
    baseline.scale[_GLOBAL_BIN] = global_scale

    if bin_column is not None and bin_column in nominal_df.columns:
        for bin_value, group in nominal_df.groupby(bin_column):
            if len(group) < min_bin_samples:
                continue
            bin_center: dict[str, float] = {}
            bin_scale: dict[str, float] = {}
            for signal in signals:
                res_col = f"res_{signal}"
                if res_col not in group.columns:
                    continue
                center, scale = _median_mad(group[res_col].to_numpy(dtype=np.float64))
                bin_center[signal] = center
                bin_scale[signal] = scale
            baseline.center[str(bin_value)] = bin_center
            baseline.scale[str(bin_value)] = bin_scale

    return baseline
