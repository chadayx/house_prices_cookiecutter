import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from house_prices_mlops.modeling.encoding import ORDINAL_ORDERS, create_ordinal_encoder
from house_prices_mlops.modeling.pipelines import create_encoding_preprocessor


def test_orders_match_notebook_exactly():
    notebook = json.loads(Path("notebooks/02_data_preparation/07_encoding.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if "".join(c["source"]).startswith("ordinal_orders = {"))
    expected = ast.literal_eval(ast.parse(source).body[0].value)
    assert list(ORDINAL_ORDERS) == list(expected)
    assert ORDINAL_ORDERS == expected


def test_ordinal_known_unknown_missing_and_absence():
    encoder = create_ordinal_encoder({"BsmtQual": ORDINAL_ORDERS["BsmtQual"]})
    encoder.fit(pd.DataFrame({"BsmtQual": ["TA", "None", "Gd"]}))
    output = encoder.transform(pd.DataFrame({"BsmtQual": ["None", "TA", "New", np.nan]}))
    np.testing.assert_array_equal(output.ravel(), [0, 3, -1, -2])
    assert ORDINAL_ORDERS["BsmtQual"] == ["None", "Po", "Fa", "TA", "Gd", "Ex"]


def test_column_composition_preserves_numeric_index_and_original():
    X = pd.DataFrame({
        "Neighborhood": ["A", "B", None],
        "BsmtQual": ["TA", "None", pd.NA],
        "OverallQual": [5, 6, 7], "OverallCond": [4, 5, 6], "LotArea": [100, 200, 300],
    }, index=[10, 20, 30])
    before = X.copy(deep=True)
    processor = create_encoding_preprocessor(nominal_columns=["Neighborhood"], ordinal_columns=["BsmtQual"])
    encoded = processor.fit_transform(X)
    np.testing.assert_array_equal(encoded["ordinal__BsmtQual"], [3, 0, -2])
    pd.testing.assert_index_equal(encoded.index, X.index)
    for column in ["OverallQual", "OverallCond", "LotArea"]:
        np.testing.assert_array_equal(encoded[f"remainder__{column}"], X[column])
    pd.testing.assert_frame_equal(X, before)
    assert processor.get_feature_names_out().tolist() == list(encoded.columns)
    validation = X.iloc[[0]].copy()
    validation["Neighborhood"] = "Unknown"
    validation["BsmtQual"] = "Unknown"
    output = processor.transform(validation)
    assert output.filter(like="onehot__").sum(axis=1).iloc[0] == 0
    assert output["ordinal__BsmtQual"].iloc[0] == -1
    onehot = processor.named_transformers_["onehot"]["encoder"]
    assert onehot.drop is None and onehot.handle_unknown == "ignore"
    assert "Unknown" not in onehot.categories_[0]


def test_selected_subset_and_pipeline():
    X = pd.DataFrame({"Neighborhood": ["A", "B", "A", "B"],
                      "ExterQual": ["TA", "Gd", "TA", "Ex"], "LotArea": [1, 2, 3, 4]})
    processor = create_encoding_preprocessor(nominal_columns=["Neighborhood"], ordinal_columns=["ExterQual"])
    model = Pipeline([("encoding", processor), ("model", Ridge())])
    cloned = clone(model)
    cloned.fit(X, [1, 2, 3, 4])
    assert len(cloned.predict(X)) == 4
    assert "Utilities" not in processor.transformers[-1][-1]
    assert not hasattr(processor, "transformers_")


@pytest.mark.parametrize("nominal,ordinal", [(["ExterQual"], ["ExterQual"]),
                                            ([], ["MissingOrder"]), (["OverallQual"], [])])
def test_invalid_encoding_configuration(nominal, ordinal):
    with pytest.raises(ValueError):
        create_encoding_preprocessor(nominal_columns=nominal, ordinal_columns=ordinal)


def test_missing_required_column_not_silently_skipped():
    processor = create_encoding_preprocessor(nominal_columns=[], ordinal_columns=["Utilities"])
    with pytest.raises(ValueError):
        processor.fit_transform(pd.DataFrame({"LotArea": [1, 2]}))
