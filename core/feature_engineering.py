"""Feature engineering helpers for forecasting models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from alphalens_forecast.utils.timeseries import (
    build_timeseries,
    series_to_dataframe,
    timeseries_to_dataframe,
)


@dataclass
class FeatureBundle:
    """Container with prepared datasets for downstream models."""

    target: pd.Series
    regressors: Optional[pd.DataFrame]
    normalized_target: pd.Series
    normalization_params: Tuple[float, float]


def zscore(series: pd.Series) -> Tuple[pd.Series, Tuple[float, float]]:
    """Return a z-scored series and the (mean, std) tuple used for scaling."""
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std == 0 or np.isnan(std):
        std = 1.0
    normalized = (series - mean) / std
    return normalized, (mean, std)


def prepare_features(
    price_frame: pd.DataFrame,
    target_column: str = "close",
) -> FeatureBundle:
    """
    Prepare target and auxiliary regressors for forecasting models.

    Parameters
    ----------
    price_frame:
        Clean price dataframe returned by the data client.
    target_column:
        Column to use as the mean-model target, defaults to the close price.

    Returns
    -------
    FeatureBundle
        Normalized target prepared for univariate training (regressors omitted).
    """
    if target_column not in price_frame.columns:
        raise KeyError(f"Target column '{target_column}' missing from price frame.")

    target = price_frame[target_column].astype(float)
    normalized_target, params = zscore(target)

    return FeatureBundle(
        target=target,
        regressors=None,
        normalized_target=normalized_target,
        normalization_params=params,
    )


def to_prophet_frame(series: pd.Series) -> pd.DataFrame:
    """Convert a target series into Prophet's expected dataframe."""
    df = series.reset_index().rename(columns={"index": "ds", series.name: "y"})
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df["y"] = df["y"].astype(float)
    return df


def to_neural_prophet_frame(
    series: pd.Series,
    regressors: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return a NeuralProphet-compatible dataframe built from the close price."""
    if regressors is not None and not regressors.empty:
        raise ValueError("NeuralProphet regressors are no longer supported.")
    frame = series_to_dataframe(series)
    ts = build_timeseries(frame)
    result = timeseries_to_dataframe(ts, value_column="y")
    result = result.rename(columns={"datetime": "ds"})
    return result


def reconstruct_from_zscore(score: float, params: Tuple[float, float]) -> float:
    """Inverse z-score scaling."""
    mean, std = params
    return score * std + mean


def compute_residuals(actual: pd.Series, fitted: pd.Series) -> pd.Series:
    """Return residual series after aligning indices."""
    fitted = fitted.reindex(actual.index).fillna(method="ffill")
    return actual - fitted
