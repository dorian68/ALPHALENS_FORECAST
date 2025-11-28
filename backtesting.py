"""Helpers for trajectory exports and simple backtesting diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ForecastTrajectory:
    """Store step-by-step predictions for a forecast horizon."""

    horizon_label: str
    timestamps: List[pd.Timestamp]
    predictions: List[float]

    def to_series(self) -> pd.Series:
        """Return predictions as a pandas Series."""
        return pd.Series(self.predictions, index=pd.DatetimeIndex(self.timestamps))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise trajectory."""
        return {
            "horizon": self.horizon_label,
            "steps": len(self.predictions),
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "predictions": self.predictions,
        }


class TrajectoryRecorder:
    """Collect trajectories during forecasting for export or backtesting."""

    def __init__(self) -> None:
        self._trajectories: List[ForecastTrajectory] = []

    def add_from_dataframe(
        self,
        horizon_label: str,
        forecast_df: pd.DataFrame,
    ) -> None:
        """Store the sequence of predictions for a given horizon."""
        if "yhat" not in forecast_df.columns:
            raise ValueError("forecast_df must contain a 'yhat' column.")
        series = forecast_df["yhat"].astype(float)
        timestamps = pd.to_datetime(series.index)
        self._trajectories.append(
            ForecastTrajectory(
                horizon_label=horizon_label,
                timestamps=list(timestamps),
                predictions=series.to_numpy(dtype=float).tolist(),
            )
        )

    def to_payload(self) -> List[Dict[str, Any]]:
        """Return serialisable trajectories."""
        return [traj.to_dict() for traj in self._trajectories]

    @property
    def trajectories(self) -> Iterable[ForecastTrajectory]:
        return tuple(self._trajectories)


def evaluate_trajectory(actual: pd.Series, trajectory: ForecastTrajectory) -> Dict[str, float]:
    """Compare a trajectory to realised prices."""
    actual_aligned, predicted = _align_series(actual, trajectory.to_series())
    errors = actual_aligned - predicted
    return {
        "rmse": float(np.sqrt(np.nanmean(np.square(errors)))),
        "mae": float(np.nanmean(np.abs(errors))),
        "direction_accuracy": _direction_accuracy(actual_aligned, predicted),
    }


def _align_series(actual: pd.Series, predicted: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = actual.index.intersection(predicted.index)
    if idx.empty:
        raise ValueError("No overlapping timestamps between actual and predicted series.")
    return actual.reindex(idx), predicted.reindex(idx)


def _direction_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    actual_diff = actual.diff().to_numpy(dtype=float)[1:]
    predicted_diff = predicted.diff().to_numpy(dtype=float)[1:]
    valid = np.logical_and(np.isfinite(actual_diff), np.isfinite(predicted_diff))
    if not valid.any():
        return float("nan")
    return float(np.mean(np.sign(actual_diff[valid]) == np.sign(predicted_diff[valid])))


__all__ = ["ForecastTrajectory", "TrajectoryRecorder", "evaluate_trajectory"]
