"""Central conceptual variable classifications, independent of tracking records."""

from collections.abc import Iterable
from contextlib import contextmanager
import json
import os
from pathlib import Path

DEFAULT_VARIABLE_TYPES = Path(__file__).resolve().parents[1] / "configs" / "variable_types.json"
VARIABLE_KINDS = {
    "continuous",
    "discrete",
    "ordinal",
    "nominal",
    "temporal",
    "identifier",
    "binary",
}


def load_variable_types(
    columns: Iterable[str] | None = None,
    path: Path = DEFAULT_VARIABLE_TYPES,
) -> dict[str, str]:
    """Read explicit metadata; optionally require exact agreement with columns.

    Never infer classifications from pandas or NumPy dtypes.
    """

    def unique_mapping(pairs):
        mapping = {}
        for name, kind in pairs:
            if name in mapping:
                raise ValueError(f"Variable duplicada en metadatos: {name}")
            mapping[name] = kind
        return mapping

    with Path(path).open(encoding="utf-8-sig") as handle:
        mapping = json.load(handle, object_pairs_hook=unique_mapping)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Los metadatos deben ser un objeto variable: clasificación no vacío.")
    for name, kind in mapping.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(kind, str)
            or kind not in VARIABLE_KINDS
        ):
            raise ValueError(f"Clasificación conceptual inválida: {name!r}: {kind!r}")
    if columns is not None:
        names = set(columns)
        unknown = set(mapping) - names
        missing = names - set(mapping)
        if unknown or missing:
            raise ValueError(
                f"Metadatos incompatibles. Variables sin columna: {sorted(unknown)}; "
                f"columnas sin clasificación: {sorted(missing)}"
            )
    return mapping


def get_variables_by_type(kind: str, columns: Iterable[str] | None = None) -> list[str]:
    """Return variables explicitly classified with the requested conceptual kind."""
    if kind not in VARIABLE_KINDS:
        raise ValueError(f"Clasificación desconocida: {kind}")
    return [name for name, value in load_variable_types(columns).items() if value == kind]


@contextmanager
def _exclusive_write(path):
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Otra escritura está en curso: {path}") from error
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink()


def save_variable_types(tracker, path=DEFAULT_VARIABLE_TYPES, *, approved=False, overwrite=False):
    """Explicitly persist approved conceptual classifications, never profile.

    Replacing different existing metadata requires overwrite=True; stale
    trackers cannot overwrite classifications approved by another notebook.
    """
    from house_prices_mlops.tracking.storage import _atomic_write

    if not approved:
        raise ValueError("Confirme las clasificaciones con approved=True.")
    if set(tracker.variable_types) != set(tracker.profile["var"]):
        raise ValueError("Las clasificaciones deben corresponder exactamente a las columnas.")
    if any(kind not in VARIABLE_KINDS for kind in tracker.variable_types.values()):
        raise ValueError("Clasificación conceptual inválida.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_write(path):
        current = load_variable_types(path=path) if path.exists() else None
        if current == tracker.variable_types:
            tracker._metadata_baseline = dict(current)
            return
        if current is not None:
            if not overwrite:
                raise FileExistsError(
                    "Los metadatos difieren; use overwrite=True tras revisarlos."
                )
            if current != tracker._metadata_baseline:
                raise RuntimeError(
                    "Los metadatos cambiaron; recargue el tracker antes de aprobar."
                )
        _atomic_write(path, (json.dumps(tracker.variable_types, indent=4) + "\n").encode("utf-8"))
        tracker._metadata_baseline = dict(tracker.variable_types)
