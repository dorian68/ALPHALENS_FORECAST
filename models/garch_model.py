"""EGARCH volatility modeling utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)


@dataclass
class EGARCHForecast:
    """Container for EGARCH variance forecasts."""

    sigma: pd.Series
    variance: pd.Series
    dof: float
    skew: float
    method: str = "analytic"


class EGARCHVolModel:
    """Wrap the arch package EGARCH(1,1) model with Student-t distribution."""

    def __init__(self) -> None:
        self._result = None
        self._residual_std: float = 0.0

    def fit(self, residuals: pd.Series) -> None:
        """Fit EGARCH(1,1) with Student-t innovations."""
        clean_residuals = residuals.dropna()
        if clean_residuals.empty:
            raise ValueError("Residual series is empty; cannot fit EGARCH.")
        std = float(clean_residuals.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            std = 1e-3
        self._residual_std = std

        model = arch_model(
            clean_residuals,
            vol="EGARCH",
            p=1,
            o=0,
            q=1,
            dist="studentst",
            rescale=False,
        )
        self._result = model.fit(disp="off")

    def forecast(
        self,
        steps: int,
    ) -> EGARCHForecast:
        """Forecast conditional variance for the requested number of steps."""
        if self._result is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        if steps <= 0:
            raise ValueError("Forecast steps must be positive.")

        last_sigma = self._last_conditional_sigma()
        attempts: Iterable[Dict[str, Optional[str]]] = (
            {"name": "analytic", "method": None},
            {"name": "simulation", "method": "simulation"},
            {"name": "bootstrap", "method": "bootstrap"},
        )

        for attempt in attempts:
            try:
                forecast = self._dispatch_forecast(steps, attempt["method"])
                variance = self._sanitize_variance(forecast.variance.iloc[-1], last_sigma)
                with np.errstate(over="raise", invalid="raise"):
                    sigma_raw = np.sqrt(variance)
                sigma = self._sanitize_sigma(sigma_raw, last_sigma)
                logger.debug(
                    "EGARCH forecast succeeded using %s method | sigma_range=(%.6f, %.6f)",
                    attempt["name"],
                    float(sigma.min()),
                    float(sigma.max()),
                )
                distribution = getattr(self._result, "distribution", None)
                dof = getattr(distribution, "nu", 6.0) if distribution is not None else 6.0
                skew = getattr(distribution, "skew", 0.0) if distribution is not None else 0.0
                return EGARCHForecast(
                    sigma=sigma,
                    variance=variance,
                    dof=float(dof),
                    skew=float(skew),
                    method=attempt["name"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "EGARCH forecast attempt using %s method failed: %s",
                    attempt["name"],
                    exc,
                )

        logger.warning(
            "EGARCH failed; using constant variance (clipped) with last sigma %.6f",
            last_sigma,
        )
        variance_fallback = pd.Series(
            np.full(steps, max(last_sigma**2, 1e-12)),
            index=pd.RangeIndex(start=1, stop=steps + 1),
            dtype=float,
        )
        sigma_fallback = pd.Series(
            np.full(steps, max(last_sigma, 1e-6)),
            index=variance_fallback.index,
            dtype=float,
        )
        distribution = getattr(self._result, "distribution", None)
        dof = getattr(distribution, "nu", 6.0) if distribution is not None else 6.0
        skew = getattr(distribution, "skew", 0.0) if distribution is not None else 0.0
        return EGARCHForecast(
            sigma=sigma_fallback,
            variance=variance_fallback,
            dof=float(dof),
            skew=float(skew),
            method="fallback_constant",
        )

    def _dispatch_forecast(self, steps: int, method: Optional[str]):
        """Call arch's forecast with optional method override."""
        kwargs = {"horizon": steps, "reindex": False}
        if method is not None:
            kwargs["method"] = method
            if method in {"simulation", "bootstrap"}:
                kwargs["simulations"] = max(1000, steps * 50)
                kwargs["random_state"] = 42
        return self._result.forecast(**kwargs)

    def _sanitize_variance(self, variance: pd.Series, fallback_sigma: float) -> pd.Series:
        """Clean variance series from the arch forecast output."""
        clean = variance.astype(float).replace([np.inf, -np.inf], np.nan)
        if clean.isna().all():
            raise RuntimeError("Variance series contains only invalid values.")
        clean = clean.ffill().bfill()
        nominal_std = self._residual_std if self._residual_std > 0 else fallback_sigma
        max_variance = max((nominal_std * 10.0) ** 2, fallback_sigma**2, 1e-8)
        clean = clean.clip(lower=1e-12, upper=max_variance)
        return clean

    def _sanitize_sigma(self, sigma: pd.Series, fallback_sigma: float) -> pd.Series:
        """Ensure sigma path is finite and reasonable."""
        clean = sigma.astype(float).replace([np.inf, -np.inf], np.nan)
        clean = clean.ffill().bfill()
        std_cap = self._residual_std if self._residual_std > 0 else fallback_sigma
        upper_bound = max(std_cap * 10.0, fallback_sigma, 1e-6)
        clean = clean.clip(lower=1e-8, upper=upper_bound)
        if clean.isna().any():
            raise RuntimeError("Sigma path contains non-finite values after sanitisation.")
        return clean

    def _last_conditional_sigma(self) -> float:
        """Return the last conditional volatility from the fitted model."""
        cond_vol = getattr(self._result, "conditional_volatility", None)
        if cond_vol is None or len(cond_vol) == 0:
            return 1e-2
        last_sigma = float(cond_vol.iloc[-1])
        if not np.isfinite(last_sigma) or last_sigma <= 0:
            return 1e-2
        return last_sigma
