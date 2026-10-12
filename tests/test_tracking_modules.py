import pandas as pd
import pytest

from house_prices_mlops.data_tracker import DataTracker as LegacyDataTracker
from house_prices_mlops.tracking import DataTracker
from house_prices_mlops.tracking.data_tracker import DataTracker as ModularDataTracker


def test_import_paths_expose_same_class():
    assert LegacyDataTracker is DataTracker is ModularDataTracker


def test_key_roundtrip_resolution_and_duplicate(tmp_path):
    data = pd.DataFrame({"Id": [1]})
    tracker = DataTracker.load(data, tmp_path)
    assert (
        tracker.add_finding(key="id_review", var="Id", stage="DU", finding="Review") == "id_review"
    )
    frame = tracker.findings
    tracker.resolve_finding(key="id_review", resolution="Reviewed")
    assert tracker.findings is frame
    assert tracker.add_finding(key="id_review", var="Id", stage="DU", finding="Review") is None
    assert tracker.findings.loc[0, "resolution"] == "Reviewed"
    tracker.save(tmp_path)
    restored = DataTracker.load(data, tmp_path)
    pd.testing.assert_frame_equal(tracker.findings, restored.findings)
    assert "id" not in restored.findings
    with pytest.raises(ValueError, match="inexistente"):
        restored.resolve_finding(key="unknown", resolution="Unknown")
    assert len(restored.findings) == 1


@pytest.mark.parametrize(
    "field,value", [("var", "Other"), ("stage", "Other"), ("finding", "Changed")]
)
def test_conflicting_key_preserves_record(field, value):
    tracker = DataTracker(pd.DataFrame({"Id": [1]}))
    values = dict(key="review", var="Id", stage="DU", finding="Review")
    tracker.add_finding(**values)
    before = tracker.findings.copy(deep=True)
    values[field] = value
    with pytest.raises(ValueError, match="Conflicto"):
        tracker.add_finding(**values)
    pd.testing.assert_frame_equal(before, tracker.findings)


@pytest.mark.parametrize("key", [None, "", "  ", 1])
def test_invalid_key(key):
    tracker = DataTracker(pd.DataFrame({"Id": [1]}))
    with pytest.raises(ValueError, match="key"):
        tracker.add_finding(key=key, var="Id", stage="DU", finding="Review")
    with pytest.raises(ValueError, match="key"):
        tracker.resolve_finding(key=key, resolution="Reviewed")


def test_old_selectors_removed():
    tracker = DataTracker(pd.DataFrame({"Id": [1]}))
    with pytest.raises(TypeError):
        tracker.add_finding("Id", "DU", "Review")
    with pytest.raises(TypeError):
        tracker.resolve_finding(1, "Reviewed")
    with pytest.raises(TypeError):
        tracker.resolve_finding(var="Id", stage="DU", finding="Review", resolution="Reviewed")


def test_pending_copy():
    tracker = DataTracker(pd.DataFrame({"Id": [1]}))
    tracker.add_finding(key="review", var="Id", stage="DU", finding="Review")
    selected = tracker.pending("DU")
    selected.loc[0, "status"] = "resolved"
    assert tracker.findings.loc[0, "status"] == "pending"
