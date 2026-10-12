import json

import pandas as pd
import pytest

from house_prices_mlops.data_tracker import DataTracker
from house_prices_mlops.metadata import load_variable_types


def test_metadata_matches_original_and_dataset():
    from house_prices_mlops.pipeline import load_stage
    import nbformat
    from pathlib import Path

    data = load_stage("train")
    mapping = load_variable_types(data.columns)
    nb = nbformat.read("notebooks/00_house_prices_original.ipynb", as_version=4)
    namespace = {"tracker": DataTracker(data, variable_types={})}
    cell = next(c for c in nb.cells if c.cell_type == "code" and "continuous_vars = [" in c.source)
    exec(cell.source, namespace)
    assert mapping == namespace["tracker"].variable_types
    assert len(data) == 1168
    assert not Path("data/interim/tracking/variable_types.json").exists()


def test_validation_rejects_missing_and_unknown_columns():
    mapping = load_variable_types()
    with pytest.raises(ValueError, match="Metadatos incompatibles"):
        load_variable_types(list(mapping)[1:])
    with pytest.raises(ValueError, match="Metadatos incompatibles"):
        load_variable_types([*mapping, "Unknown"])


def test_tracker_preserves_records_without_saving_metadata(tmp_path):
    data = pd.DataFrame({"Id": [1], "SalePrice": [100]})
    tracker = DataTracker.load(data, tmp_path)
    finding_key = tracker.add_finding(key='review', var="Id", stage="DU", finding="Review")
    tracker.resolve_finding(key=finding_key, resolution="Reviewed")
    tracker.log("Id", "DU", "review", "manual")
    tracker.add_plan("Id", "DU", "exclude", "identifier")
    tracker.save(tmp_path)
    assert not (tmp_path / "variable_types.json").exists()
    restored = DataTracker.load(data, tmp_path)
    restored.add_finding(key='review', var="Id", stage="DU", finding="Review")
    assert len(restored.findings) == 1
    assert restored.findings.iloc[0]["resolution"] == "Reviewed"
    assert len(restored.history) == 1 and len(restored.plans) == 1
    assert restored.variable_types == load_variable_types()


def test_stale_tracker_cannot_overwrite_records(tmp_path):
    data = pd.DataFrame({"Id": [1], "SalePrice": [100]})
    first = DataTracker.load(data, tmp_path)
    stale = DataTracker.load(data, tmp_path)
    first.add_finding(key='keep', var="Id", stage="DU", finding="Keep")
    first.save(tmp_path)
    before = (tmp_path / "findings.json").read_bytes()
    with pytest.raises(RuntimeError, match="recargue"):
        stale.save(tmp_path)
    with pytest.raises(RuntimeError, match="recargue"):
        DataTracker(data).save(tmp_path)
    assert (tmp_path / "findings.json").read_bytes() == before
    restored = DataTracker.load(data, tmp_path)
    restored.add_finding(key='keep', var="Id", stage="DU", finding="Keep")
    restored.log("Id", "DU", "review")
    restored.log("Id", "DU", "review")
    restored.add_plan("Id", "DU", "drop", "identifier")
    restored.add_plan("Id", "DU", "drop", "identifier")
    assert len(restored.findings) == len(restored.history) == len(restored.plans) == 1
    restored.save(tmp_path)
    assert not (tmp_path / "profile.json").exists()


def test_explicit_approval_and_metadata_conflicts(tmp_path):
    mapping = load_variable_types()
    data = pd.DataFrame({name: [1] for name in mapping})
    tracker = DataTracker(data)
    path = tmp_path / "variable_types.json"
    with pytest.raises(ValueError, match="approved=True"):
        tracker.save_variable_types(path)
    tracker.save_variable_types(path, approved=True)
    assert load_variable_types(path=path) == mapping
    original = path.read_bytes()
    tracker.save_variable_types(path, approved=True)
    assert path.read_bytes() == original
    stale = DataTracker(data)
    tracker.set_variable_types(["Electrical"], "nominal")
    with pytest.raises(FileExistsError):
        tracker.save_variable_types(path, approved=True)
    tracker.save_variable_types(path, approved=True, overwrite=True)
    assert load_variable_types(path=path)["Electrical"] == "nominal"
    stale.set_variable_types(["Fence"], "nominal")
    with pytest.raises(RuntimeError, match="recargue"):
        stale.save_variable_types(path, approved=True, overwrite=True)


def test_profile_is_rebuilt_from_current_values(tmp_path):
    first = DataTracker.load(pd.DataFrame({"Id": [1], "SalePrice": [100]}), tmp_path)
    first.save(tmp_path)
    second = DataTracker.load(pd.DataFrame({"Id": [2], "SalePrice": [999]}), tmp_path)
    assert second.profile.set_index("var").loc["SalePrice", "mean"] == 999


def test_default_tracking_path_is_independent_of_working_directory(tmp_path, monkeypatch):
    import inspect
    from house_prices_mlops.config import INTERIM_DATA_DIR
    from house_prices_mlops.data_tracker import DEFAULT_TRACKING_DIR

    monkeypatch.chdir(tmp_path)
    expected = INTERIM_DATA_DIR / "tracking"
    assert DEFAULT_TRACKING_DIR == expected and expected.is_absolute()
    for method in (DataTracker.load, DataTracker.save):
        assert inspect.signature(method).parameters["path"].default == expected
    data = pd.DataFrame({"Id": [1], "SalePrice": [100]})
    default = DataTracker.load(data)
    explicit = DataTracker.load(data, expected)
    pd.testing.assert_frame_equal(default.findings, explicit.findings)
    pd.testing.assert_frame_equal(default.history, explicit.history)
    pd.testing.assert_frame_equal(default.plans, explicit.plans)
