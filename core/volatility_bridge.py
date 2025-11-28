"""Helpers to bridge mean forecasts with GARCH volatility outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class VolatilityForecast:
    """Container for volatility projections."""

    horizon_steps: int
    conditional_variance: pd.Series
    sigma_path: pd.Series
    dof: float


def horizon_to_steps(horizon_hours: int, interval: str) -> int:
    """Convert a time horizon in hours to number of bars given the sampling interval."""
    interval = interval.lower()
    mapping: Dict[str, float] = {
        "1min": 1 / 60,
        "5min": 5 / 60,
        "15min": 15 / 60,
        "30min": 0.5,
        "45min": 0.75,
        "1h": 1.0,
        "2h": 2.0,
        "3h": 3.0,
        "4h": 4.0,
        "6h": 6.0,
        "8h": 8.0,
        "12h": 12.0,
        "1day": 24.0,
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported interval '{interval}' for horizon conversion.")
    hours_per_step = mapping[interval]
    steps = int(np.ceil(horizon_hours / hours_per_step))
    return max(steps, 1)


def interval_to_hours(interval: str) -> float:
    """Convert a Twelve Data interval string to hours."""
    mapping: Dict[str, float] = {
        "1min": 1 / 60,
        "5min": 5 / 60,
        "15min": 15 / 60,
        "30min": 0.5,
        "45min": 0.75,
        "1h": 1.0,
        "2h": 2.0,
        "3h": 3.0,
        "4h": 4.0,
        "6h": 6.0,
        "8h": 8.0,
        "12h": 12.0,
        "1day": 24.0,
    }
    try:
        return mapping[interval.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported interval '{interval}'") from exc


def prepare_residuals(residuals: pd.Series) -> pd.Series:
    """Ensure residuals are centered and drop missing values."""
    clean = residuals.dropna()
    clean = clean - clean.mean()
    return clean


def annualize_volatility(sigma: float, periods_per_year: int) -> float:
    """Annualize a per-period volatility."""
    return sigma * np.sqrt(periods_per_year)


def deannualize_volatility(annual_vol: float, periods_per_year: int) -> float:
    """Convert annualized volatility to per-period."""
    return annual_vol / np.sqrt(periods_per_year)
