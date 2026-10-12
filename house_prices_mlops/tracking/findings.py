"""Finding registration, resolution and queries using explicit stable keys."""

import pandas as pd


def _validate_key(key):
    if not isinstance(key, str) or not key.strip():
        raise ValueError("La key debe ser un texto no vacío.")


def add_finding(findings, *, key, var, stage, finding):
    _validate_key(key)
    matches = findings.loc[findings["key"].eq(key)]
    if len(matches) > 1:
        raise ValueError(f"Key duplicada: {key}")
    if not matches.empty:
        existing = matches.iloc[0]
        if any(
            existing[field] != value
            for field, value in (("var", var), ("stage", stage), ("finding", finding))
        ):
            raise ValueError(f"Conflicto de contenido para key: {key}")
        return findings, None
    record = {
        "key": key,
        "var": var,
        "stage": stage,
        "finding": finding,
        "status": "pending",
        "resolution": None,
    }
    return pd.concat([findings, pd.DataFrame([record])], ignore_index=True), key


def pending(findings, stage=None):
    result = findings[findings["status"] == "pending"]
    if stage is not None:
        result = result[result["stage"] == stage]
    return result.copy()


def resolve_finding(findings, *, key, resolution):
    _validate_key(key)
    mask = findings["key"].eq(key)
    if not mask.any():
        raise ValueError(f"Hallazgo inexistente para key: {key}")
    if mask.sum() > 1:
        raise ValueError(f"Key duplicada: {key}")
    findings["resolution"] = findings["resolution"].astype("object")
    findings.loc[mask, "status"] = "resolved"
    findings.loc[mask, "resolution"] = resolution
