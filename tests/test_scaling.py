import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from house_prices_mlops.modeling.pipelines import create_scaling_preprocessor
from house_prices_mlops.modeling.scaling import create_scaler


@pytest.mark.parametrize("method,cls", [("standard", StandardScaler), ("robust", RobustScaler),
                                       ("minmax", MinMaxScaler)])
def test_scaler_matches_sklearn_and_preserves_data(method, cls):
    train = pd.DataFrame({"area": [1.0, 2.0, 4.0, np.nan]})
    test = pd.DataFrame({"area": [100.0, np.nan]}, index=[10, 20])
    original = train.copy(deep=True)
    scaler = create_scaler(method).fit(train)
    result = scaler.transform(test)
    expected = cls().fit(train).transform(test)
    np.testing.assert_allclose(result, expected, equal_nan=True)
    np.testing.assert_allclose(scaler.inverse_transform(result), test, equal_nan=True)
    pd.testing.assert_frame_equal(train, original)
    assert np.isnan(result[1, 0])
    if method == "minmax":
        assert result[0, 0] > 1  # Test does not change training bounds or clip by default.


def test_composition_keeps_other_columns_index_and_names():
    X = pd.DataFrame({"area": [1.0, 2.0, 3.0], "HasPool": [0, 1, 0],
                      "Neighborhood": ["A", "B", "C"]}, index=[10, 20, 30])
    before = X.copy(deep=True)
    processor = create_scaling_preprocessor(columns=["area"], method="standard")
    result = processor.fit_transform(X)
    np.testing.assert_allclose(result["scaling__area"], StandardScaler().fit_transform(X[["area"]]).ravel())
    pd.testing.assert_index_equal(result.index, X.index)
    for column in ["HasPool", "Neighborhood"]:
        np.testing.assert_array_equal(result[f"remainder__{column}"], X[column])
    assert list(result.columns) == processor.get_feature_names_out().tolist()
    pd.testing.assert_frame_equal(X, before)


def test_pipeline_cloning_and_fold_specific_statistics():
    X = pd.DataFrame({"area": np.arange(20, dtype=float)})
    y = np.arange(20, dtype=float) * 2
    model = Pipeline([
        ("scaling", create_scaling_preprocessor(columns=["area"], method="standard")),
        ("model", Ridge()),
    ])
    cloned = clone(model)
    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    results = cross_validate(cloned, X, y, cv=cv, return_estimator=True, error_score="raise")
    for fitted, (train, test) in zip(results["estimator"], cv.split(X)):
        scaler = fitted["scaling"].named_transformers_["scaling"]
        np.testing.assert_allclose(scaler.mean_, X.iloc[train].mean())
        assert len(fitted.predict(X.iloc[test])) == len(test)
    assert not hasattr(cloned["scaling"], "transformers_")


@pytest.mark.parametrize("columns", [[], ["Id"], ["SalePrice"], ["area", "area"]])
def test_invalid_scaling_columns(columns):
    with pytest.raises(ValueError):
        create_scaling_preprocessor(columns=columns, method="standard")


def test_explicit_method_parameters_and_invalid_method():
    with pytest.raises(TypeError):
        create_scaler()
    with pytest.raises(ValueError, match="desconocido"):
        create_scaler("model_dependent")
    with pytest.raises(ValueError, match="copy=True"):
        create_scaler("standard", copy=False)
    assert create_scaler("minmax", feature_range=(-1, 1)).feature_range == (-1, 1)
