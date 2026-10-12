import pandas as pd
import pytest

from house_prices_mlops.dataset import (
    get_predictors_and_target,
    get_test_data,
    get_training_data,
    load_split_manifest,
)


def test_stable_partition_and_predictors(tmp_path):
    source, manifest = tmp_path / "train.csv", tmp_path / "split.csv"
    data = pd.DataFrame(
        {"Id": range(20, 0, -1), "SalePrice": range(100, 120), "LotArea": range(200, 220)}
    )
    data.to_csv(source, index=False)
    load_split_manifest(source, manifest)
    original = manifest.read_bytes()
    train, test = get_training_data(source, manifest), get_test_data(source, manifest)
    assert len(train) == 16 and len(test) == 4
    assert set(train.Id).isdisjoint(test.Id)
    assert set(train.Id) | set(test.Id) == set(data.Id)
    data.sample(frac=1, random_state=7).to_csv(source, index=False)
    pd.testing.assert_frame_equal(train, get_training_data(source, manifest))
    assert manifest.read_bytes() == original
    other = tmp_path / "other.csv"
    load_split_manifest(source, other)
    assert other.read_bytes() == original
    X, y = get_predictors_and_target(train)
    assert list(X.columns) == ["LotArea"]
    pd.testing.assert_series_equal(y, train.SalePrice)


@pytest.mark.parametrize("change", ["value", "add", "remove", "corrupt"])
def test_changed_data_or_manifest_rejected(tmp_path, change):
    source, path = tmp_path / "train.csv", tmp_path / "split.csv"
    data = pd.DataFrame({"Id": range(10), "SalePrice": range(100, 110)})
    data.to_csv(source, index=False)
    _, manifest = load_split_manifest(source, path)
    if change == "value":
        data.loc[0, "SalePrice"] += 1
    elif change == "add":
        data.loc[10] = [10, 110]
    elif change == "remove":
        data = data.iloc[1:]
    else:
        manifest.loc[0, "split"] = "invalid"
        manifest.to_csv(path, index=False)
    data.to_csv(source, index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="No se sobrescribió"):
        load_split_manifest(source, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("problem", ["missing_id", "missing_target", "null", "duplicate"])
def test_invalid_input(tmp_path, problem):
    source, path = tmp_path / "train.csv", tmp_path / "split.csv"
    data = pd.DataFrame({"Id": [1, 2, 3], "SalePrice": [100, 200, 300]})
    if problem == "missing_id":
        data = data.drop(columns="Id")
    elif problem == "missing_target":
        data = data.drop(columns="SalePrice")
    elif problem == "null":
        data.loc[0, "Id"] = None
    else:
        data.loc[0, "Id"] = 2
    data.to_csv(source, index=False)
    with pytest.raises(ValueError):
        load_split_manifest(source, path)
    assert not path.exists()
