import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate

from house_prices_mlops.modeling.feature_selection import fit_low_variability_selection
from house_prices_mlops.modeling.pipelines import build_model_pipeline
from house_prices_mlops.modeling.scaling import create_scaler


@pytest.fixture
def data():
    index = pd.Index(range(20, 32), name="housing")
    X = pd.DataFrame(
        {
            "LotArea": np.arange(1, 13) * 10.0,
            "LotFrontage": [1, np.nan, 4, 9, 16, 25, np.nan, 36, 49, 64, 81, 100],
            "TotalSF": np.arange(12) * 17.0 + 100,
            "all_missing": [np.nan] * 12,
            "Utilities": ["AllPub"] * 12,
            "constant_nominal": ["A"] * 12,
            "OverallQual": [5, 6, 7] * 4,
            "OverallCond": [4, 5, 6] * 4,
            "ExterQual": ["TA", "Gd", "Ex"] * 4,
            "Neighborhood": [f"N{i}" for i in range(12)],
            "MasVnrType": ["None", None, "BrkFace"] * 4,
        },
        index=index,
    )
    y = pd.Series(np.exp(np.linspace(0, 4, 12)) * 100, index=index, name="SalePrice")
    return X, y


@pytest.mark.parametrize("scaling", [None, "standard", "robust", "minmax"])
@pytest.mark.parametrize("selection", [False, True])
def test_integrated_preparation_prediction_and_no_mutation(data, scaling, selection):
    X, y = data
    before = X.copy(deep=True)
    model = build_model_pipeline(
        Ridge(),
        use_selection=selection,
        max_target_association=0.1,
        scaling=scaling,
        log_target=True,
    )
    model.fit(X, y)
    output = model[:-1].transform(X)
    assert np.isfinite(output.to_numpy()).all()
    pd.testing.assert_index_equal(output.index, X.index)
    assert any("TotalSF" in name for name in output)
    assert len(model.predict(X)) == len(X)
    if selection:
        assert "Utilities" in model["selection"].drop_columns_
        assert "constant_nominal" in model["selection"].drop_columns_
        assert not any("Utilities" in name for name in output)
        assert not any("constant_nominal" in name for name in output)
    else:
        assert any("Utilities" in name for name in output)
    validation = X.iloc[[0]].copy()
    validation["Neighborhood"] = "Unknown"
    validation["ExterQual"] = "Unknown"
    assert np.isfinite(model.predict(validation)).all()
    pd.testing.assert_frame_equal(X, before)


@pytest.mark.parametrize("scaling", ["standard", "robust", "minmax"])
def test_cv_fits_selection_imputation_categories_scaling_on_training_only(data, scaling):
    X, y = data
    model = build_model_pipeline(
        Ridge(),
        use_selection=True,
        max_target_association=0.1,
        scaling=scaling,
        log_target=True,
    )
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    result = cross_validate(model, X, y, cv=cv, return_estimator=True, error_score="raise")
    for fitted, (train, validation) in zip(result["estimator"], cv.split(X)):
        X_train, y_train = X.iloc[train], y.iloc[train]
        expected_drops = fit_low_variability_selection(
            X_train, y_train, max_target_association=0.1
        )
        assert fitted["selection"].drop_columns_ == expected_drops
        selected = X_train.drop(columns=expected_drops)
        numeric = fitted["preprocessing"].named_transformers_["numeric"]
        columns = list(numeric["imputer"].feature_names_in_)
        np.testing.assert_allclose(
            numeric["imputer"].statistics_,
            selected[columns].median().fillna(0),
        )
        imputed = numeric["imputer"].transform(selected[columns])
        transformed = numeric["distributions"].transform(imputed)
        reference = create_scaler(scaling).fit(transformed)
        np.testing.assert_allclose(
            numeric["scaler"].transform(transformed), reference.transform(transformed)
        )
        nominal = fitted["preprocessing"].named_transformers_["nominal"]["encoder"]
        position = list(nominal.feature_names_in_).index("Neighborhood")
        assert set(nominal.categories_[position]) == set(X_train["Neighborhood"])
        assert set(nominal.categories_[position]).isdisjoint(X.iloc[validation]["Neighborhood"])
        assert len(fitted.predict(X.iloc[validation])) == len(validation)
    assert not hasattr(model["preprocessing"], "transformers_")


def test_selector_receives_original_target_before_final_log(data, monkeypatch):
    import house_prices_mlops.modeling.feature_selection as selection_module

    X, y = data
    original = selection_module.fit_low_variability_selection
    seen = []

    def capture(X, y, **params):
        seen.append(y.copy())
        return original(X, y, **params)

    monkeypatch.setattr(selection_module, "fit_low_variability_selection", capture)
    model = build_model_pipeline(
        Ridge(), use_selection=True, max_target_association=0.1, log_target=True
    )
    model.fit(X, y)
    pd.testing.assert_series_equal(seen[0], y)
    transformed = model["estimator"].transformer_.transform(y.to_numpy().reshape(-1, 1)).ravel()
    np.testing.assert_allclose(transformed, np.log(y))
    seen.clear()
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    cross_validate(model, X, y, cv=cv, error_score="raise")
    assert len(seen) == 3
    for actual, (train, _) in zip(seen, cv.split(X)):
        pd.testing.assert_series_equal(actual, y.iloc[train])


def test_selection_threshold_and_predictor_validation(data):
    X, y = data
    with pytest.raises(ValueError, match="max_target_association"):
        build_model_pipeline(Ridge(), use_selection=True)
    with pytest.raises(ValueError, match="Id y SalePrice"):
        build_model_pipeline(Ridge()).fit(X.assign(SalePrice=y), y)


@pytest.mark.parametrize("scaling", [None, "standard", "robust", "minmax"])
@pytest.mark.parametrize("selection", [False, True])
@pytest.mark.parametrize("log_target", [False, True])
def test_common_dataset_full_pipeline_keeps_source_unchanged(scaling, selection, log_target):
    from house_prices_mlops.dataset import get_predictors_and_target
    from house_prices_mlops.pipeline import load_stage

    source = load_stage("common")
    before = source.copy(deep=True)
    X, y = get_predictors_and_target(source)
    model = build_model_pipeline(
        Ridge(),
        use_selection=selection,
        max_target_association=0.1,
        scaling=scaling,
        log_target=log_target,
    )
    model.fit(X, y)
    prediction = model.predict(X.iloc[:10])
    assert prediction.shape == (10,)
    assert np.isfinite(prediction).all()
    prepared = model[:-1].transform(X)
    assert len(prepared) == len(source) == 1168
    assert np.isfinite(prepared.to_numpy()).all()
    pd.testing.assert_index_equal(prepared.index, source.index)
    pd.testing.assert_frame_equal(source, before)
    pd.testing.assert_frame_equal(source, load_stage("common"))
    result = cross_validate(
        build_model_pipeline(
            Ridge(),
            use_selection=selection,
            max_target_association=0.1,
            scaling=scaling,
            log_target=log_target,
        ),
        X,
        y,
        cv=KFold(n_splits=2, shuffle=True, random_state=42),
        return_estimator=True,
        error_score="raise",
    )
    for fitted in result["estimator"]:
        assert np.isfinite(fitted.predict(X.iloc[:10])).all()


def test_constructor_is_unfitted_cloneable_and_exposes_nested_parameters():
    estimator = Ridge(alpha=2)
    model = build_model_pipeline(estimator, scaling="standard", log_target=True)
    copied = clone(model)
    assert list(copied.named_steps) == ["predictors", "selection", "preprocessing", "estimator"]
    assert not hasattr(model["preprocessing"], "transformers_")
    assert not hasattr(estimator, "coef_")
    copied.set_params(
        estimator__regressor__alpha=3, preprocessing__numeric__scaler__with_mean=False
    )
    assert copied.get_params()["estimator__regressor__alpha"] == 3
    assert model.get_params()["estimator__regressor__alpha"] == 2


def test_numeric_scales_do_not_affect_nominal_or_ordinal_outputs(data):
    X, y = data
    unscaled = build_model_pipeline(Ridge(), scaling=None).fit(X, y)
    scaled = build_model_pipeline(Ridge(), scaling="standard").fit(X, y)
    first, second = unscaled[:-1].transform(X), scaled[:-1].transform(X)
    categorical = [column for column in first if not column.startswith("numeric__")]
    pd.testing.assert_frame_equal(first[categorical], second[categorical])
    numeric = scaled["preprocessing"].named_transformers_["numeric"]
    assert "OverallQual" in numeric["imputer"].feature_names_in_
    assert "OverallCond" in numeric["imputer"].feature_names_in_
    np.testing.assert_array_equal(first["numeric__OverallQual"], X["OverallQual"])
    np.testing.assert_array_equal(first["numeric__OverallCond"], X["OverallCond"])


def test_preprocessor_reexports_are_the_same_functions():
    from house_prices_mlops.modeling import distributions, encoding, pipelines, scaling

    assert pipelines.create_distribution_preprocessor is distributions.create_distribution_preprocessor
    assert pipelines.create_encoding_preprocessor is encoding.create_encoding_preprocessor
    assert pipelines.create_scaling_preprocessor is scaling.create_scaling_preprocessor


def test_ridge_script_cv_keeps_recorded_results_without_mlflow_writes():
    import runpy

    script = runpy.run_path("scripts/train.py")
    X, y = script["load_training_data"]()
    model = script["build_model"]()
    folds, summary, metrics = script["evaluate_model"](model, X, y)
    assert model["selection"] == "passthrough"
    assert model["estimator"].__class__.__name__ == "Ridge"
    assert len(folds) == 5
    expected = {
        "mae_mean": 19829.00638429834,
        "mae_std": 1587.0130971652936,
        "rmse_mean": 37105.18961176624,
        "rmse_std": 11831.546404820594,
        "r2_mean": 0.7548479785486731,
        "r2_std": 0.1449279907256722,
    }
    for metric, value in expected.items():
        assert metrics[metric] == pytest.approx(value, rel=1e-8)


def test_analysis_notebook_has_no_training_or_mlflow_writes():
    import ast
    import json
    from pathlib import Path

    notebook = json.loads(Path("notebooks/03_modeling/01_experiments.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                assert name not in {"fit", "fit_transform", "cross_validate", "start_run", "log_params", "log_metrics"}
