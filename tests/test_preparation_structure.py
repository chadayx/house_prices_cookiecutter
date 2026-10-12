import pandas as pd
import pytest

from house_prices_mlops import pipeline
from house_prices_mlops.preparation.structure import normalize_ms_subclass, prepare_structure


def test_structure_preserves_original_rows_and_other_columns():
    data = pd.DataFrame({"Id": [2, 1], "MSSubClass": [20, 90], "SalePrice": [100, 200]})
    before = data.copy(deep=True)
    result = prepare_structure(data)
    assert result is not data
    assert result["MSSubClass"].tolist() == ["20", "90"]
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(result.drop(columns="MSSubClass"), before.drop(columns="MSSubClass"))
    pd.testing.assert_frame_equal(result, prepare_structure(data))
    pd.testing.assert_frame_equal(result, prepare_structure(result))


@pytest.mark.parametrize("transform", [normalize_ms_subclass, prepare_structure])
def test_nullable_string_preserves_missing_values(transform):
    data = pd.DataFrame({"Id": [1, 2, 3], "MSSubClass": [20, None, pd.NA]})
    before = data.copy(deep=True)
    result = transform(data)
    assert result is not data
    assert str(result["MSSubClass"].dtype) == "string"
    pd.testing.assert_series_equal(result["MSSubClass"].isna(), before["MSSubClass"].isna())
    assert result["MSSubClass"].iloc[0] == "20"
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(result, normalize_ms_subclass(data))


@pytest.mark.parametrize("split", ["train", "test"])
def test_integrated_structure_matches_individual_transform(monkeypatch, split):
    data = pd.DataFrame({"Id": [1, 2], "MSSubClass": [20, 90], "SalePrice": [100, 200]})
    before = data.copy(deep=True)
    monkeypatch.setattr(pipeline, "get_training_data", lambda: data)
    monkeypatch.setattr(pipeline, "get_test_data", lambda: data)
    result = pipeline.load_stage("structure", split=split)
    pd.testing.assert_frame_equal(result, normalize_ms_subclass(data))
    pd.testing.assert_frame_equal(result, pipeline.load_stage("structure", split=split))
    pd.testing.assert_frame_equal(data, before)
