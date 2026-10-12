import pandas as pd
import pytest

from house_prices_mlops import pipeline
from house_prices_mlops.dataset import get_test_data, get_training_data, load_split_manifest


@pytest.fixture
def partitions(tmp_path, monkeypatch):
    source = tmp_path / "train.csv"
    manifest = tmp_path / "split_manifest.csv"
    pd.DataFrame({"Id": range(20), "SalePrice": range(100, 120), "value": range(20)}).to_csv(
        source, index=False
    )
    raw, _ = load_split_manifest(source, manifest)
    train = get_training_data(source, manifest)
    test = get_test_data(source, manifest)
    monkeypatch.setattr(pipeline, "load_split_manifest", lambda: (raw, None))
    monkeypatch.setattr(pipeline, "get_training_data", lambda: train)
    monkeypatch.setattr(pipeline, "get_test_data", lambda: test)
    return raw, train, test, tmp_path


def test_legacy_states(partitions):
    for name, frame in zip(("raw", "train", "test"), partitions[:3]):
        pd.testing.assert_frame_equal(pipeline.load_stage(name), frame)
        pd.testing.assert_frame_equal(pipeline.load_stage(name, split="test"), frame)


def add_one(data):
    data = data.copy()
    data["value"] += 1
    return data


def double(data):
    data = data.copy()
    data["value"] *= 2
    return data


@pytest.mark.parametrize("split,position", [("train", 1), ("test", 2)])
def test_prepared_states_order_reconstruction_and_no_mutation(partitions, monkeypatch, split, position):
    monkeypatch.setattr(
        pipeline, "PREPARATION_STEPS", (("added", add_one), ("doubled", double))
    )
    original = partitions[position]
    before = original.copy(deep=True)
    files = {p.name: p.read_bytes() for p in partitions[3].iterdir()}
    first = pipeline.load_stage("added", split=split)
    final = pipeline.load_stage("doubled", split=split)
    pd.testing.assert_series_equal(first["value"], before["value"] + 1)
    pd.testing.assert_series_equal(final["value"], (before["value"] + 1) * 2)
    pd.testing.assert_frame_equal(final, pipeline.load_stage("doubled", split=split))
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(final[["Id", "SalePrice"]], before[["Id", "SalePrice"]])
    assert len(final) == len(before)
    assert files == {p.name: p.read_bytes() for p in partitions[3].iterdir()}
    if split == "train":
        pd.testing.assert_frame_equal(final, pipeline.load_stage("doubled"))


@pytest.mark.parametrize("stage", ["unknown", "unimplemented", "structural_clean"])
def test_unregistered_states_rejected_before_loading(monkeypatch, stage):
    def unexpected_load():
        pytest.fail("Unknown states must not load data")

    monkeypatch.setattr(pipeline, "get_training_data", unexpected_load)
    with pytest.raises(ValueError, match="Etapa desconocida"):
        pipeline.load_stage(stage)


def test_invalid_prepared_split(partitions, monkeypatch):
    monkeypatch.setattr(pipeline, "PREPARATION_STEPS", (("added", add_one),))
    with pytest.raises(ValueError, match="Partición desconocida"):
        pipeline.load_stage("added", split="raw")


@pytest.mark.parametrize("split,position", [("train", 1), ("test", 2)])
def test_stage_groups_functions_and_stops_before_next_stage(partitions, monkeypatch, split, position):
    def unexpected_transform(data):
        pytest.fail("Later stages must not run")

    monkeypatch.setattr(
        pipeline,
        "PREPARATION_STEPS",
        (("grouped", lambda data: double(add_one(data))), ("later", unexpected_transform)),
    )
    before = partitions[position].copy(deep=True)
    result = pipeline.load_stage("grouped", split=split)
    pd.testing.assert_series_equal(result["value"], (before["value"] + 1) * 2)
    pd.testing.assert_frame_equal(result, pipeline.load_stage("grouped", split=split))
    pd.testing.assert_frame_equal(partitions[position], before)
    pd.testing.assert_frame_equal(result[["Id", "SalePrice"]], before[["Id", "SalePrice"]])


@pytest.mark.parametrize("transform", [add_one, double])
def test_transform_contract_returns_new_frame(partitions, transform):
    original = partitions[1]
    before = original.copy(deep=True)
    result = transform(original)
    assert result is not original
    pd.testing.assert_frame_equal(original, before)


@pytest.mark.parametrize("split", ["train", "test"])
def test_missing_matches_original_categorical_fill_and_preserves_veneer_unknowns(monkeypatch, split):
    from house_prices_mlops.preparation.missing import (
        normalize_masonry_absence,
        normalize_structural_absence,
        prepare_missing,
    )

    columns = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtQual", "BsmtCond",
    ]
    data = pd.DataFrame({column: [None, None, "Existing"] for column in columns})
    for column in ("PoolArea", "Fireplaces", "GarageCars", "GarageArea", "TotalBsmtSF"):
        data[column] = [0, 1, 0]
    data["Id"] = [1, 2, 3]
    data["SalePrice"] = [100, 200, 300]
    data["MSSubClass"] = [20, 30, 90]
    data["MasVnrType"] = [None, None, None]
    data["MasVnrArea"] = [0, 10, None]
    data["GarageYrBlt"] = [None, 2000, None]
    data["LotFrontage"] = [None, 10, None]
    data["Electrical"] = [None, "SBrkr", None]
    original = data.copy(deep=True)
    for transform in (normalize_structural_absence, normalize_masonry_absence, prepare_missing):
        result = transform(data)
        assert result is not data
        pd.testing.assert_frame_equal(data, original)
        pd.testing.assert_frame_equal(result, transform(result))
    result = prepare_missing(data)
    assert result.loc[0, "GarageType"] == "None"
    assert result.loc[1, "GarageType"] == "None"
    assert result.loc[2, "GarageType"] == "Existing"
    assert result.loc[0, "MasVnrType"] == "None"
    assert pd.isna(result.loc[1, "MasVnrType"])
    assert pd.isna(result.loc[2, "MasVnrType"])
    assert result.loc[0, "PoolQC"] == "None"
    assert result.loc[1, "PoolQC"] == "None"
    unchanged = ["Id", "SalePrice", "GarageYrBlt", "MasVnrArea", "LotFrontage", "Electrical"]
    pd.testing.assert_frame_equal(result[unchanged], original[unchanged])
    monkeypatch.setattr(pipeline, "get_training_data", lambda: data)
    monkeypatch.setattr(pipeline, "get_test_data", lambda: data)
    integrated = pipeline.load_stage("missing", split=split)
    pd.testing.assert_frame_equal(integrated, prepare_missing(pipeline.load_stage("structure", split=split)))
    pd.testing.assert_frame_equal(integrated, pipeline.load_stage("missing", split=split))
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("split", ["train", "test"])
def test_common_features_match_original_formulas_and_preserve_source(split):
    from house_prices_mlops.preparation.feature_engineering import (
        create_aggregate_features,
        create_presence_features,
        create_quality_area_interaction,
        create_temporal_features,
        normalize_remodel_age,
        prepare_features,
    )

    before = pipeline.load_stage("missing", split=split)
    original = before.copy(deep=True)
    temporal = create_temporal_features(before)
    for transform, data in (
        (create_temporal_features, before),
        (normalize_remodel_age, temporal),
        (create_aggregate_features, before),
        (create_presence_features, before),
        (create_quality_area_interaction, before),
        (prepare_features, before),
    ):
        snapshot = data.copy(deep=True)
        assert transform(data) is not data
        pd.testing.assert_frame_equal(data, snapshot)
    result = pipeline.load_stage("common", split=split)
    pd.testing.assert_frame_equal(result, prepare_features(before))
    pd.testing.assert_frame_equal(result[before.columns], original)
    pd.testing.assert_frame_equal(result, pipeline.load_stage("common", split=split))
    pd.testing.assert_series_equal(
        result["TotalSF"], before["TotalBsmtSF"] + before["GrLivArea"], check_names=False
    )
    pd.testing.assert_series_equal(
        result["RemodAge"], (before["YrSold"] - before["YearRemodAdd"]).clip(lower=0),
        check_names=False,
    )
    assert result.loc[before["GarageYrBlt"].isna(), "GarageAge"].isna().all()
    assert set(result["HasGarage"].unique()) <= {0, 1}


@pytest.mark.parametrize("split,expected_rows", [("train", 1168), ("test", 292)])
def test_all_common_stages_preserve_partition_without_target_dependency(monkeypatch, split, expected_rows):
    original = pipeline.load_stage(split)
    snapshot = original.copy(deep=True)
    for stage in ("structure", "missing", "common"):
        prepared = pipeline.load_stage(stage, split=split)
        assert len(prepared) == expected_rows
        pd.testing.assert_series_equal(prepared["Id"], original["Id"])
        pd.testing.assert_series_equal(prepared["SalePrice"], original["SalePrice"])
        pd.testing.assert_frame_equal(original, snapshot)

    expected = pipeline.load_stage("common", split=split).drop(columns="SalePrice")
    without_target = original.drop(columns="SalePrice")
    loader = "get_training_data" if split == "train" else "get_test_data"
    monkeypatch.setattr(pipeline, loader, lambda: without_target)
    pd.testing.assert_frame_equal(pipeline.load_stage("common", split=split), expected)
    pd.testing.assert_frame_equal(without_target, snapshot.drop(columns="SalePrice"))
    # A row's result must not depend on statistics of the surrounding batch.
    single_row = without_target.iloc[[0]].copy()
    monkeypatch.setattr(pipeline, loader, lambda: single_row)
    pd.testing.assert_frame_equal(pipeline.load_stage("common", split=split), expected.iloc[[0]])
