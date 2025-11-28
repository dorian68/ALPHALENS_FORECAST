"""Persistence utilities for AlphaLens forecasting artifacts."""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class StoredArtifacts:
    """Container for saved model artifacts and metadata."""

    model_path: Path
    manifest_path: Path
    metadata: Dict[str, Any]
    payload: Optional[Dict[str, Any]] = None
    mean_model: Optional[Any] = None
    vol_model: Optional[Any] = None
    timestamp_slug: Optional[str] = None


class ModelStore:
    """Filesystem-backed storage for trained models and manifests."""

    def __init__(self, base_dir: Path, logger: logging.Logger) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger

    def save(
        self,
        prefix: str,
        mean_model: Any,
        vol_model: Any,
        metadata: Dict[str, Any],
        payload: Optional[Dict[str, Any]],
    ) -> StoredArtifacts:
        """Persist models, metadata, and payload to disk."""
        model_path = self.base_dir / f"{prefix}.pkl"
        manifest_path = self.base_dir / f"{prefix}.json"

        with open(model_path, "wb") as handle:
            pickle.dump(
                {
                    "mean_model": mean_model,
                    "vol_model": vol_model,
                },
                handle,
            )

        manifest = {
            "metadata": metadata,
            "payload": payload,
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        return StoredArtifacts(
            model_path=model_path,
            manifest_path=manifest_path,
            metadata=metadata,
            payload=payload,
            timestamp_slug=metadata.get("timestamp_slug"),
        )

    def load_latest(self, symbol_slug: str, timeframe_slug: str) -> Optional[StoredArtifacts]:
        """Load the most recent artifacts for a given symbol/timeframe combination."""
        pattern = f"{symbol_slug}_{timeframe_slug}_*.json"
        manifests = list(self.base_dir.glob(pattern))
        if not manifests:
            self._logger.debug(
                "No stored artifacts found for symbol=%s timeframe=%s in %s",
                symbol_slug,
                timeframe_slug,
                self.base_dir,
            )
            return None

        latest_manifest_path = max(manifests, key=lambda path: path.stat().st_mtime)
        prefix = latest_manifest_path.stem
        model_path = self.base_dir / f"{prefix}.pkl"

        try:
            with open(latest_manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.warning("Failed to load manifest %s: %s", latest_manifest_path, exc)
            return None

        metadata = manifest.get("metadata", {})
        payload = manifest.get("payload")
        mean_model = None
        vol_model = None

        if model_path.exists():
            try:
                with open(model_path, "rb") as handle:
                    stored_models = pickle.load(handle)
                mean_model = stored_models.get("mean_model")
                vol_model = stored_models.get("vol_model")
            except (OSError, pickle.UnpicklingError) as exc:
                self._logger.warning("Failed to load model pickle %s: %s", model_path, exc)
        else:
            self._logger.warning("Model pickle missing for manifest %s", latest_manifest_path)

        artifacts = StoredArtifacts(
            model_path=model_path,
            manifest_path=latest_manifest_path,
            metadata=metadata,
            payload=payload,
            mean_model=mean_model,
            vol_model=vol_model,
            timestamp_slug=metadata.get("timestamp_slug"),
        )

        self._logger.info(
            "Loaded stored artifacts %s (hash=%s)",
            latest_manifest_path,
            metadata.get("data_hash"),
        )
        return artifacts


__all__ = ["ModelStore", "StoredArtifacts"]
