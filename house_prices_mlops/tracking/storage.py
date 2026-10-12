import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from house_prices_mlops.config import INTERIM_DATA_DIR

DEFAULT_TRACKING_DIR = INTERIM_DATA_DIR / "tracking"

_RECORD_COLUMNS = {
    "findings": ["key", "var", "stage", "finding", "status", "resolution"],
    "history": ["id", "var", "stage", "action", "method", "before", "after"],
    "plans": ["id", "var", "stage", "action", "method", "params", "status"],
}


def _atomic_write(path, content):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record_snapshot(path):
    path = Path(path)
    return {
        name: (path / name).read_bytes() if (path / name).exists() else None
        for name in ("findings.json", "history.json", "plans.json")
    }


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _json_bytes(frame, kind):
    records = frame.to_dict(orient="records")
    ordered = []
    ids = set()
    for row in records:
        field = "key" if kind == "findings" else "id"
        identifier = row.get(field)
        if kind == "findings":
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"Key inválida en findings: {identifier!r}")
            if "id" in row:
                raise ValueError("Los hallazgos deben identificarse exclusivamente mediante key.")
        else:
            if isinstance(identifier, bool) or not isinstance(identifier, (int, np.integer)):
                raise TypeError(f"ID inválido en {kind}: {identifier!r}")
            if identifier < 1:
                raise ValueError(f"ID inválido en {kind}: {identifier}")
        if identifier in ids:
            raise ValueError(f"{field} repetida en {kind}: {identifier}")
        ids.add(identifier)
        fields = _RECORD_COLUMNS[kind] + sorted(set(row) - set(_RECORD_COLUMNS[kind]))
        ordered.append({key: _json_value(row[key]) for key in fields if key in row})
    ordered.sort(key=lambda row: row["key" if kind == "findings" else "id"])
    return (json.dumps(ordered, ensure_ascii=False, indent=4, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _invalid_constant(value):
    raise ValueError(f"Constante JSON inválida: {value}")


def _read_records(path, kind):
    file = Path(path) / f"{kind}.json"
    if not file.exists():
        return pd.DataFrame(columns=_RECORD_COLUMNS[kind])
    records = json.loads(file.read_text(encoding="utf-8-sig"), parse_constant=_invalid_constant)
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"Formato inválido: {file}")
    columns = _RECORD_COLUMNS[kind] + sorted(
        {key for row in records for key in row} - set(_RECORD_COLUMNS[kind])
    )
    frame = pd.DataFrame(records, columns=columns)
    _json_bytes(frame, kind)
    return frame


def write_records(path, frames, expected):
    """Validate everything before writing; refuse changed files, no batch transaction."""
    path = Path(path)
    if _record_snapshot(path) != expected:
        raise RuntimeError(
            "Los registros cambiaron o ya existen; recargue con DataTracker.load()."
        )
    payload = {f"{kind}.json": _json_bytes(frame, kind) for kind, frame in frames.items()}
    path.mkdir(parents=True, exist_ok=True)
    for name, content in payload.items():
        current = (path / name).read_bytes() if (path / name).exists() else None
        if current != expected[name]:
            raise RuntimeError("Los registros cambiaron; recargue con DataTracker.load().")
        if current != content:
            _atomic_write(path / name, content)
