"""NeuralProphet forecaster wrapper."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from neuralprophet import NeuralProphet
from pandas.tseries.frequencies import to_offset

from alphalens_forecast.core.feature_engineering import to_neural_prophet_frame
from alphalens_forecast.models.base import BaseForecaster


class NeuralProphetForecaster(BaseForecaster):
    """Wrapper around NeuralProphet configured for intraday forecasting."""

    def __init__(self) -> None:
        super().__init__(name="NeuralProphet")
        self._model: Optional[NeuralProphet] = None
        self._train_frame: Optional[pd.DataFrame] = None
        self._freq: Optional[str] = None

    def fit(
        self,
        target: pd.Series,
        regressors: Optional[pd.DataFrame] = None,
    ) -> None:
        frame = to_neural_prophet_frame(target, regressors)
        freq = pd.infer_freq(pd.DatetimeIndex(frame["ds"]))
        if freq is None:
            deltas = frame["ds"].diff().dropna()
            if deltas.empty:
                raise ValueError("Unable to infer data frequency for NeuralProphet.")
            freq = to_offset(deltas.mode().iloc[0]).freqstr
        self._freq = freq

        model = NeuralProphet(
            n_lags=0,
            n_forecasts=1,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
            learning_rate=0.01,
        )

        self._model = model
        self._train_frame = frame
        self._model.fit(frame, freq=freq, progress=None)

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if self._model is None or self._train_frame is None:
            raise RuntimeError("NeuralProphetForecaster must be fitted first.")
        if future_regressors is not None and not future_regressors.empty:
            raise ValueError("NeuralProphet forecasts no longer accept future regressors.")

        future = self._model.make_future_dataframe(
            self._train_frame,
            periods=steps,
            n_historic_predictions=False,
        )

        forecast = self._model.predict(future)
        return forecast[["ds", "yhat1"]].rename(columns={"yhat1": "yhat"})
