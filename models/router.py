"""Model routing utilities."""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from alphalens_forecast.utils.text import slugify

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Persist and load models using the convention ``models/{type}/{symbol}/{tf}``.

    This provides a single choke point so both training jobs and CLI inference
    store assets identically. Metadata is stored alongside the pickle to help
    orchestration layers reason about freshness.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base_dir = Path(base_dir or Path("models")).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def get_model_dir(self, model_type: str, symbol: str, timeframe: str) -> Path:
        """Return the directory that hosts the pickle/manifest for the model."""
        return (
            self._base_dir
            / model_type.lower()
            / slugify(symbol)
            / slugify(timeframe)
        )

    def get_model_path(self, model_type: str, symbol: str, timeframe: str) -> Path:
        return self.get_model_dir(model_type, symbol, timeframe) / "model.pkl"

    def get_metadata_path(self, model_type: str, symbol: str, timeframe: str) -> Path:
        return self.get_model_dir(model_type, symbol, timeframe) / "metadata.json"

    def save_model(
        self,
        model_type: str,
        symbol: str,
        timeframe: str,
        model: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist a fitted model and optional metadata."""
        model_dir = self.get_model_dir(model_type, symbol, timeframe)
        model_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_artifacts"):
            try:
                model.save_artifacts(model_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Model %s for %s @ %s failed to save auxiliary artifacts: %s",
                    model_type,
                    symbol,
                    timeframe,
                    exc,
                )
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as handle:
            pickle.dump(model, handle)
        manifest = {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_type": model_type,
            "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "metadata": metadata or {},
        }
        with open(self.get_metadata_path(model_type, symbol, timeframe), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        logger.info("Saved %s model for %s @ %s to %s", model_type, symbol, timeframe, model_path)

    def load_model(self, model_type: str, symbol: str, timeframe: str) -> Optional[Any]:
        """Return the model if it exists, otherwise None."""
        model_path = self.get_model_path(model_type, symbol, timeframe)
        if not model_path.exists():
            logger.debug("No %s model found for %s @ %s", model_type, symbol, timeframe)
            return None
        try:
            with open(model_path, "rb") as handle:
                model = pickle.load(handle)
            logger.info("Loaded %s model for %s @ %s from %s", model_type, symbol, timeframe, model_path)
            if hasattr(model, "load_artifacts"):
                try:
                    model.load_artifacts(self.get_model_dir(model_type, symbol, timeframe))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Model %s for %s @ %s failed to load auxiliary artifacts: %s",
                        model_type,
                        symbol,
                        timeframe,
                        exc,
                    )
            return model
        except (OSError, pickle.UnpicklingError) as exc:
            logger.warning("Failed to load %s model for %s @ %s: %s", model_type, symbol, timeframe, exc)
            return None

    # Convenience wrappers for volatility models -------------------------

    def save_egarch(self, symbol: str, timeframe: str, model: Any, metadata: Optional[dict[str, Any]] = None) -> None:
        """Persist EGARCH results under the dedicated namespace."""
        self.save_model("egarch", symbol, timeframe, model, metadata)

    def load_egarch(self, symbol: str, timeframe: str) -> Optional[Any]:
        """Return the EGARCH model if available."""
        return self.load_model("egarch", symbol, timeframe)


__all__ = ["ModelRouter"]
