"""Offline training entrypoints for AlphaLens models."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import pandas as pd

from alphalens_forecast.core import prepare_features, prepare_residuals
from alphalens_forecast.data import DataProvider
from alphalens_forecast.models import EGARCHVolModel
from alphalens_forecast.models.base import BaseForecaster
from alphalens_forecast.models.router import ModelRouter
from alphalens_forecast.models.selection import instantiate_model
from alphalens_forecast.training_schedule import TRAINING_FREQUENCIES

logger = logging.getLogger(__name__)


def _default_provider(provider: Optional[DataProvider]) -> DataProvider:
    return provider or DataProvider()


def _default_router(router: Optional[ModelRouter]) -> ModelRouter:
    return router or ModelRouter()


def train_mean_model(
    model_type: str,
    symbol: str,
    timeframe: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
) -> BaseForecaster:
    """Shared training loop for Prophet/NeuralProphet/NHiTS backends."""
    provider = _default_provider(data_provider)
    router = _default_router(model_router)
    frame = price_frame if price_frame is not None else provider.load_data(symbol, timeframe)
    features = prepare_features(frame)
    model = instantiate_model(model_type)
    model.fit(features.target, features.regressors)
    metadata = {
        "n_observations": len(frame),
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_frequency": TRAINING_FREQUENCIES[model_type]["frequency"],
    }
    router.save_model(model_type, symbol, timeframe, model, metadata=metadata)
    return model


def train_nhits(
    symbol: str,
    timeframe: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
) -> BaseForecaster:
    """Train/persist an N-HiTS mean model."""
    return train_mean_model(
        "nhits",
        symbol,
        timeframe,
        price_frame=price_frame,
        data_provider=data_provider,
        model_router=model_router,
    )


def train_neuralprophet(
    symbol: str,
    timeframe: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
) -> BaseForecaster:
    """Train/persist a NeuralProphet mean model."""
    return train_mean_model(
        "neuralprophet",
        symbol,
        timeframe,
        price_frame=price_frame,
        data_provider=data_provider,
        model_router=model_router,
    )


def train_prophet(
    symbol: str,
    timeframe: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
) -> BaseForecaster:
    """Train/persist a Prophet mean model."""
    return train_mean_model(
        "prophet",
        symbol,
        timeframe,
        price_frame=price_frame,
        data_provider=data_provider,
        model_router=model_router,
    )


def train_egarch(
    symbol: str,
    timeframe: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    residuals: Optional[pd.Series] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
) -> EGARCHVolModel:
    """Train an EGARCH model for the given book."""
    provider = _default_provider(data_provider)
    router = _default_router(model_router)
    frame = price_frame if price_frame is not None else provider.load_data(symbol, timeframe)
    resids = residuals if residuals is not None else prepare_residuals(frame["log_return"])
    model = EGARCHVolModel()
    model.fit(resids)
    metadata = {
        "n_observations": len(resids.dropna()),
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_frequency": TRAINING_FREQUENCIES["egarch"]["frequency"],
    }
    router.save_egarch(symbol, timeframe, model, metadata=metadata)
    return model


MEAN_TRAINERS: Dict[str, Callable[..., BaseForecaster]] = {
    "nhits": train_nhits,
    "neuralprophet": train_neuralprophet,
    "prophet": train_prophet,
}


__all__ = [
    "MEAN_TRAINERS",
    "train_egarch",
    "train_mean_model",
    "train_neuralprophet",
    "train_nhits",
    "train_prophet",
]
