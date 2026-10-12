import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from house_prices_mlops.modeling.pipelines import create_distribution_preprocessor
from house_prices_mlops.modeling.distributions import (
    DISTRIBUTION_TRANSFORMATIONS,
    TARGET_TRANSFORMATION,
    create_distribution_transformer,
)
from house_prices_mlops.pipeline import load_stage


@pytest.mark.parametrize("method,function", [("log", np.log), ("sqrt", np.sqrt), ("log1p", np.log1p)])
def test_stateless_transform_matches_formula_preserves_missing_and_inverse(method, function):
    data = np.array([[1.0], [4.0], [np.nan]])
    original = data.copy()
    transformer = create_distribution_transformer(method)
    result = transformer.fit_transform(data)
    np.testing.assert_allclose(result, function(data), equal_nan=True)
    np.testing.assert_allclose(transformer.inverse_transform(result), original, equal_nan=True)
    np.testing.assert_allclose(data, original, equal_nan=True)
    np.testing.assert_allclose(clone(transformer).fit_transform(data[:1]), result[:1])


def test_unknown_distribution_method():
    with pytest.raises(ValueError, match="desconocida"):
        create_distribution_transformer("boxcox")


def test_distribution_component_train_test_and_target_separation():
    train = load_stage("common")
    test = load_stage("common", split="test")
    original = train.copy(deep=True)
    transformer = create_distribution_preprocessor()
    transformer.fit(train.drop(columns=["Id", "SalePrice"]))
    for data in (train, test):
        output = pd.DataFrame(transformer.transform(data), columns=transformer.get_feature_names_out())
        assert len(output) == len(data)
        assert output.shape[1] == len(DISTRIBUTION_TRANSFORMATIONS)
        for var, method in DISTRIBUTION_TRANSFORMATIONS.items():
            function = {"log": np.log, "sqrt": np.sqrt, "log1p": np.log1p}[method]
            np.testing.assert_allclose(output[f"{method}__{var}"], function(data[var]), equal_nan=True)
        assert all("SalePrice" not in name and "Id" not in name for name in output)
    pd.testing.assert_frame_equal(train, original)
    assert TARGET_TRANSFORMATION == "log"
