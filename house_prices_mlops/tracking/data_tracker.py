"""Integration interface for profiles, records and persistence."""

from pathlib import Path

import pandas as pd

from house_prices_mlops.metadata import (
    DEFAULT_VARIABLE_TYPES,
    VARIABLE_KINDS,
    load_variable_types,
    save_variable_types,
)
from house_prices_mlops.tracking import findings, history, plans, profile, storage
from house_prices_mlops.tracking.storage import DEFAULT_TRACKING_DIR


class DataTracker:
    def __init__(self, df, variable_types=None):

        self.variable_types = (
            load_variable_types() if variable_types is None else dict(variable_types)
        )

        self._metadata_baseline = load_variable_types()
        self._records_baseline = None
        self._records_path = None

        self.profile = pd.DataFrame()

        self.findings = pd.DataFrame(
            columns=["key", "var", "stage", "finding", "status", "resolution"]
        )

        self.history = pd.DataFrame(
            columns=["id", "var", "stage", "action", "method", "before", "after"]
        )
        self.plans = pd.DataFrame(
            columns=["id", "var", "stage", "action", "method", "params", "status"]
        )
        self.update_profile(df)

    def set_variable_types(self, variables, variable_type):
        if variable_type not in VARIABLE_KINDS:
            raise ValueError(f"Clasificación desconocida: {variable_type}")
        variables = list(variables)
        unknown = set(variables) - set(self.profile["var"])
        if unknown:
            raise ValueError(f"Variables inexistentes: {sorted(unknown)}")

        for var in variables:
            self.variable_types[var] = variable_type

        self.profile["variable_type"] = self.profile["var"].map(self.variable_types)

    def update_profile(self, df):
        self.profile = profile.build_profile(df, self.variable_types)
        return self.profile

    def add_finding(self, *, key, var, stage, finding):
        self.findings, finding_key = findings.add_finding(
            self.findings, key=key, var=var, stage=stage, finding=finding
        )
        return finding_key

    def resolve_finding(self, *, key, resolution):
        findings.resolve_finding(self.findings, key=key, resolution=resolution)

    def pending(self, stage=None):
        return findings.pending(self.findings, stage)

    def add_plan(self, var, stage, action, method, params=None):
        self.plans = plans.add_plan(self.plans, var, stage, action, method, params)

    def log(self, var, stage, action, method=None, before=None, after=None):
        self.history, event_id = history.log(
            self.history, var, stage, action, method, before, after
        )
        return event_id

    def save_variable_types(self, path=DEFAULT_VARIABLE_TYPES, *, approved=False, overwrite=False):
        return save_variable_types(self, path, approved=approved, overwrite=overwrite)

    def save(self, path=DEFAULT_TRACKING_DIR):
        path = Path(path).resolve()
        current = storage._record_snapshot(path)
        expected = self._records_baseline if self._records_path == path else dict.fromkeys(current)
        storage.write_records(
            path, {kind: getattr(self, kind) for kind in storage._RECORD_COLUMNS}, expected
        )
        self._records_path = path
        self._records_baseline = storage._record_snapshot(path)

    @classmethod
    def load(cls, df, path=DEFAULT_TRACKING_DIR, variable_types=None):
        path = Path(path).resolve()
        before = storage._record_snapshot(path)
        tracker = cls(df, variable_types=variable_types)
        for kind in storage._RECORD_COLUMNS:
            frame = storage._read_records(path, kind)
            setattr(tracker, kind, frame)
        if before != storage._record_snapshot(path):
            raise RuntimeError("Los registros cambiaron durante la carga; vuelva a cargar.")
        tracker._records_path = path
        tracker._records_baseline = before
        return tracker
