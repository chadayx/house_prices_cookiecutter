import numpy as np
import pandas as pd
import pytest

from house_prices_mlops.modeling.feature_selection import (
    LowVariabilitySelector,
    analyze_low_variability,
    apply_feature_selection,
    fit_low_variability_selection,
)


@pytest.fixture
def training():
    y = pd.Series([0.0] * 95 + [1.0] * 5)
    X = pd.DataFrame({
        "constant": [1] * 100,
        "rare_relevant": y,
        "rare_category_relevant": ["A"] * 95 + ["B"] * 5,
        "category_constant": ["A"] * 100,
        "varying": np.arange(100),
        "unknown": [np.nan] * 100,
    })
    return X, y


def test_combined_criterion_retains_rare_but_relevant_features(training):
    X, y = training
    before = X.copy(deep=True)
    target = y.copy(deep=True)
    review = analyze_low_variability(X, y).set_index("var")
    assert review.loc["rare_relevant", "low_variability"]
    assert review.loc["rare_relevant", "target_association"] == pytest.approx(1)
    assert review.loc["rare_category_relevant", "target_association"] == pytest.approx(1)
    assert review.loc["rare_category_relevant", "association_method"] == "eta"
    drops = fit_low_variability_selection(X, y, max_target_association=0.1)
    assert drops == ["constant", "category_constant"]
    assert fit_low_variability_selection(X, y, max_target_association=0.1) == drops
    pd.testing.assert_frame_equal(X, before)
    pd.testing.assert_series_equal(y, target)


def test_validation_uses_training_selection_without_relearning(training):
    X, y = training
    drops = fit_low_variability_selection(X, y, max_target_association=0.1)
    validation = X.iloc[:10].copy()
    validation["constant"] = np.arange(10)
    before = validation.copy(deep=True)
    selected = apply_feature_selection(validation, drop_columns=drops)
    assert list(selected.columns) == [var for var in X if var not in drops]
    assert len(selected) == len(validation)
    pd.testing.assert_index_equal(selected.index, validation.index)
    pd.testing.assert_frame_equal(validation, before)
    pd.testing.assert_frame_equal(selected, before.drop(columns=drops))


def test_selection_must_be_relearned_when_training_target_changes(training):
    X, y = training
    alternate = pd.Series(np.tile([0.0, 1.0], 50))
    first = fit_low_variability_selection(X, y, max_target_association=0.1)
    second = fit_low_variability_selection(X, alternate, max_target_association=0.1)
    assert "rare_relevant" not in first
    assert "rare_relevant" in second


def test_explicit_threshold_and_predictor_only_input(training):
    X, y = training
    with pytest.raises(TypeError):
        fit_low_variability_selection(X, y)
    with pytest.raises(ValueError, match="Id y SalePrice"):
        analyze_low_variability(X.assign(SalePrice=y), y)
    with pytest.raises(ValueError, match="índices"):
        analyze_low_variability(X, y.iloc[::-1])
    with pytest.raises(ValueError, match="variar"):
        analyze_low_variability(X, pd.Series([1.0] * len(X)))
    with pytest.raises(ValueError, match="entre 0 y 1"):
        fit_low_variability_selection(X, y, max_target_association=2)


def test_selector_fit_transform_and_names(training):
    X, y = training
    before = X.copy(deep=True)
    selector = LowVariabilitySelector(max_target_association=0.1)
    assert selector.fit(X, y) is selector
    assert selector.drop_columns_ == fit_low_variability_selection(X, y, max_target_association=0.1)
    validation = X.iloc[:10].copy()
    validation.index = pd.Index(range(200, 210), name="housing")
    validation["constant"] = np.arange(10)
    validation_before = validation.copy(deep=True)
    result = selector.transform(validation)
    pd.testing.assert_frame_equal(result, validation.drop(columns=selector.drop_columns_))
    assert result is not validation
    assert selector.get_feature_names_out().tolist() == list(result.columns)
    assert selector.n_features_in_ == X.shape[1]
    pd.testing.assert_frame_equal(X, before)
    pd.testing.assert_frame_equal(validation, validation_before)


@pytest.mark.parametrize("change", ["missing", "extra", "reordered"])
def test_selector_rejects_different_schema(training, change):
    X, y = training
    selector = LowVariabilitySelector(max_target_association=0.1).fit(X, y)
    changed = {
        "missing": X.drop(columns="constant"),
        "extra": X.assign(extra=0),
        "reordered": X[X.columns[::-1]],
    }[change]
    with pytest.raises(ValueError, match="colonnes|columnas"):
        selector.transform(changed)


def test_selector_requires_fit_and_dataframe(training):
    from sklearn.exceptions import NotFittedError

    X, y = training
    selector = LowVariabilitySelector(max_target_association=0.1)
    with pytest.raises(NotFittedError):
        selector.transform(X)
    with pytest.raises(TypeError, match="DataFrame"):
        selector.fit(X.to_numpy(), y)
    selector.fit(X, y)
    with pytest.raises(TypeError, match="DataFrame"):
        selector.transform(X.to_numpy())


def test_selector_pipeline_clone_and_cross_validation():
    from sklearn.base import clone
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.pipeline import Pipeline

    X = pd.DataFrame({"constant": [1] * 20, "signal": np.arange(20)})
    y = pd.Series(np.arange(20) * 2.0)
    model = Pipeline([
        ("selection", LowVariabilitySelector(max_target_association=0.1)),
        ("model", Ridge()),
    ])
    cloned = clone(model)
    model.fit(X, y)
    assert len(model.predict(X)) == len(X)
    assert model["selection"].drop_columns_ == ["constant"]
    assert not hasattr(cloned["selection"], "drop_columns_")
    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    results = cross_validate(cloned, X, y, cv=cv, return_estimator=True, error_score="raise")
    for fitted, (train, _) in zip(results["estimator"], cv.split(X)):
        expected = fit_low_variability_selection(X.iloc[train], y.iloc[train], max_target_association=0.1)
        assert fitted["selection"].drop_columns_ == expected
