import json

import pandas as pd
import pytest

import house_prices_mlops.tracking.storage as module
from house_prices_mlops.data_tracker import DataTracker


@pytest.fixture
def data():
    return pd.DataFrame({"Id": [1], "SalePrice": [100]})


def test_empty_and_null_roundtrip(tmp_path, data):
    tracker = DataTracker.load(data, tmp_path)
    tracker.save(tmp_path)
    assert (tmp_path / "history.json").read_text() == "[]\n"
    tracker.add_finding(key="review", var="Id", stage="DU", finding="Review")
    tracker.save(tmp_path)
    records = json.loads((tmp_path / "findings.json").read_text())
    assert records[0]["resolution"] is None
    assert len(DataTracker.load(data, tmp_path).findings) == 1


def test_atomic_file_write_preserves_previous_json_on_failure(tmp_path, data, monkeypatch):
    tracker = DataTracker.load(data, tmp_path)
    tracker.add_finding(key="existing", var="Id", stage="DU", finding="Existing")
    tracker.save(tmp_path)
    before = (tmp_path / "findings.json").read_bytes()
    tracker.add_finding(key="new", var="Id", stage="DU", finding="New")
    original = module.os.replace

    def interrupted(source, destination):
        raise OSError("Interrupted")

    monkeypatch.setattr(module.os, "replace", interrupted)
    with pytest.raises(OSError):
        tracker.save(tmp_path)
    assert (tmp_path / "findings.json").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "findings.json",
        "history.json",
        "plans.json",
    ]
    monkeypatch.setattr(module.os, "replace", original)
    tracker.save(tmp_path)
    assert len(DataTracker.load(data, tmp_path).findings) == 2


def test_partial_batch_failure_keeps_valid_records_and_requires_reload(
    tmp_path, data, monkeypatch
):
    tracker = DataTracker.load(data, tmp_path)
    tracker.add_finding(key="existing", var="Id", stage="DU", finding="Existing")
    tracker.log("Id", "DU", "existing")
    tracker.save(tmp_path)
    tracker.add_finding(key="new", var="Id", stage="DU", finding="New")
    tracker.log("Id", "DU", "new")
    original = module._atomic_write

    def interrupted(path, content):
        if path.name == "history.json":
            raise OSError("Interrupted")
        original(path, content)

    monkeypatch.setattr(module, "_atomic_write", interrupted)
    with pytest.raises(OSError):
        tracker.save(tmp_path)
    reloaded = DataTracker.load(data, tmp_path)
    assert len(reloaded.findings) == 2
    assert len(reloaded.history) == 1
    with pytest.raises(RuntimeError, match="recargue"):
        tracker.save(tmp_path)
    monkeypatch.setattr(module, "_atomic_write", original)
    reloaded.log("Id", "DU", "new")
    reloaded.save(tmp_path)
    assert len(DataTracker.load(data, tmp_path).history) == 2


def test_nonfinite_values_and_external_edits_are_rejected(tmp_path, data):
    tracker = DataTracker.load(data, tmp_path)
    tracker.save(tmp_path)
    original = module._record_snapshot(tmp_path)
    tracker.add_plan("Id", "DU", "drop", "identifier", {"threshold": float("inf")})
    with pytest.raises(ValueError):
        tracker.save(tmp_path)
    assert module._record_snapshot(tmp_path) == original
    (tmp_path / "findings.json").write_text("[]\n\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="recargue"):
        tracker.save(tmp_path)


@pytest.mark.parametrize("content", ["{invalid", "[NaN]", "[Infinity]", "{}"])
def test_invalid_json_is_rejected(tmp_path, data, content):
    (tmp_path / "findings.json").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        DataTracker.load(data, tmp_path)


@pytest.mark.parametrize("key", [None, "", "   ", 1])
def test_invalid_persisted_key_is_rejected(tmp_path, data, key):
    rows = [
        {
            "key": key,
            "var": "Id",
            "stage": "DU",
            "finding": "Review",
            "status": "pending",
            "resolution": None,
        }
    ]
    (tmp_path / "findings.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Key inválida"):
        DataTracker.load(data, tmp_path)


def test_legacy_id_records_require_explicit_migration(tmp_path, data):
    (tmp_path / "findings.json").write_text('[{"id": 1}]', encoding="utf-8")
    with pytest.raises(ValueError, match="Key inválida"):
        DataTracker.load(data, tmp_path)


def test_json_roundtrip_preserves_ids_extra_fields_and_duplicates(tmp_path, data):
    rows = [
        {
            "key": "texto_ñ,_con_coma",
            "var": "NA",
            "stage": "DU",
            "finding": "Texto ñ, con coma",
            "status": "resolved",
            "resolution": "",
            "extra": "preservar",
        }
    ]
    (tmp_path / "findings.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tracker = DataTracker.load(data, tmp_path)
    assert tracker.findings.iloc[0]["extra"] == "preservar"
    assert (
        tracker.add_finding(
            key="texto_ñ,_con_coma", var="NA", stage="DU", finding="Texto ñ, con coma"
        )
        is None
    )
    assert len(tracker.findings) == 1
    assert tracker.add_finding(key="nuevo", var="Id", stage="DU", finding="Nuevo") == "nuevo"
    tracker.save(tmp_path)
    before = module._record_snapshot(tmp_path)
    restored = DataTracker.load(data, tmp_path)
    assert list(restored.findings["key"]) == ["nuevo", "texto_ñ,_con_coma"]
    restored.save(tmp_path)
    assert module._record_snapshot(tmp_path) == before
    (tmp_path / "findings.json").write_text(json.dumps(rows + rows), encoding="utf-8")
    with pytest.raises(ValueError, match="key repetida"):
        DataTracker.load(data, tmp_path)
