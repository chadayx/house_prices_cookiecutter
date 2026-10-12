"""Backward-compatible import for the modular DataTracker."""

from house_prices_mlops.tracking.data_tracker import DataTracker
from house_prices_mlops.tracking.storage import DEFAULT_TRACKING_DIR

__all__ = ["DEFAULT_TRACKING_DIR", "DataTracker"]
