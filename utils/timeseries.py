"""Shared helpers for univariate TimeSeries construction."""
from __future__ import annotations

import pandas as pd
from darts import TimeSeries


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean dataframe with ``datetime``/``close`` columns."""
    if "datetime" not in df.columns:
        raise KeyError("Dataframe must contain a 'datetime' column.")
    if "close" not in df.columns:
        raise KeyError("Dataframe must contain a 'close' column.")
    frame = df.copy()
    frame = frame.dropna(subset=["datetime", "close"])
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"])
    frame = frame.sort_values("datetime")
    return frame


def build_timeseries(df: pd.DataFrame) -> TimeSeries:
    """
    Build a Darts ``TimeSeries`` using only datetime/close columns.

    Parameters
    ----------
    df:
        Dataframe with ``datetime`` and ``close`` columns.
    """
    frame = _normalize_dataframe(df)
    return TimeSeries.from_dataframe(frame, time_col="datetime", value_cols="close")


def series_to_dataframe(series: pd.Series) -> pd.DataFrame:
    """Convert a ``pd.Series`` indexed by datetime into a two-column frame."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Target series must have a DatetimeIndex.")
    values = pd.to_numeric(series.astype(float), errors="coerce")
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(values.index, utc=True),
            "close": values.values,
        }
    )
    frame = frame.dropna(subset=["close"])
    frame = frame.sort_values("datetime")
    return frame


def timeseries_to_dataframe(series: TimeSeries, value_column: str) -> pd.DataFrame:
    """Convert a Darts ``TimeSeries`` back into a dataframe."""
    if hasattr(series, "pd_dataframe"):
        df = series.pd_dataframe()
    else:
        df = series.to_dataframe()
    columns = df.columns
    first_column = columns[0] if columns.nlevels == 1 else columns[0]
    values = df.loc[:, [first_column]].copy()
    values.columns = [value_column]
    values = values.reset_index()
    values = values.rename(columns={values.columns[0]: "datetime"})
    values["datetime"] = pd.to_datetime(values["datetime"], utc=True)
    return values


__all__ = ["build_timeseries", "series_to_dataframe", "timeseries_to_dataframe"]
