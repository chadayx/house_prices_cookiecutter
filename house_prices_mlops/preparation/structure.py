"""Validación de estructura y tipos de variables."""

import pandas as pd


def normalize_ms_subclass(data: pd.DataFrame) -> pd.DataFrame:
    """Represent MSSubClass codes as text, preserving the original data."""
    result = data.copy()
    result["MSSubClass"] = result["MSSubClass"].astype("string")
    return result


def prepare_structure(data: pd.DataFrame) -> pd.DataFrame:
    """Apply approved structural transformations in order."""
    return normalize_ms_subclass(data)
