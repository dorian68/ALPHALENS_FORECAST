"""Caching data provider that abstracts Twelve Data fetches."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from alphalens_forecast.config import TwelveDataConfig
from alphalens_forecast.utils import TwelveDataClient
from alphalens_forecast.utils.text import slugify

logger = logging.getLogger(__name__)


class DataProvider:
    """
    Central access point for price data with simple filesystem caching.

    The provider keeps a per-(symbol, timeframe) CSV under ``data/cache`` so
    repeated forecasts reuse local history instead of hammering the API. The
    implementation is storage-agnostic and can be swapped for S3/GCS later
    by overriding ``_read_cache``/``_write_cache``.
    """

    def __init__(
        self,
        config: Optional[TwelveDataConfig] = None,
        cache_dir: Optional[Path] = None,
        client: Optional[TwelveDataClient] = None,
    ) -> None:
        self._config = config or TwelveDataConfig()
        self._cache_dir = Path(cache_dir or Path("data") / "cache").resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or TwelveDataClient(self._config)

    def get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Return the cache path for a (symbol, timeframe) pair."""
        symbol_slug = slugify(symbol)
        timeframe_slug = slugify(timeframe)
        return self._cache_dir / symbol_slug / f"{timeframe_slug}.csv"

    def load_data(self, symbol: str, timeframe: str, refresh: bool = False) -> pd.DataFrame:
        """
        Return a historical dataframe, using cache when possible.

        Parameters
        ----------
        symbol, timeframe:
            Series identifier.
        refresh:
            Force a fresh download even if cache exists.
        """
        cache_path = self.get_cache_path(symbol, timeframe)
        if not refresh:
            cached = self._read_cache(cache_path)
            if cached is not None:
                logger.debug("Serving %s @ %s from cache", symbol, timeframe)
                return cached
        return self.load_latest(symbol, timeframe, persist=True)

    def load_latest(self, symbol: str, timeframe: str, persist: bool = True) -> pd.DataFrame:
        """
        Download the latest history from the upstream provider.

        When ``persist`` is True the cache is updated after merging with any
        stored history so repeated calls only append new data.
        """
        frame = self._client.fetch_ohlcv(symbol=symbol, interval=timeframe)
        cache_path = self.get_cache_path(symbol, timeframe)
        cached = self._read_cache(cache_path)
        if cached is not None:
            combined = pd.concat([cached, frame]).sort_index()
            frame = combined[~combined.index.duplicated(keep="last")]
        if persist:
            self._write_cache(cache_path, frame)
        return frame

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, parse_dates=["datetime"], infer_datetime_format=True)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read cache %s: %s; ignoring cache.", path, exc)
            return None
        df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def _write_cache(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = frame.copy()
        serialisable.index.name = "datetime"
        try:
            serialisable.to_csv(path)
        except OSError as exc:
            logger.warning("Failed to write cache %s: %s", path, exc)


__all__ = ["DataProvider"]
