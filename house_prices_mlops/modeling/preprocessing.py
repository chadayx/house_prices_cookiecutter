"""Ramas de preprocesamiento y resolución dinámica de columnas tras la selección."""

from functools import partial

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from house_prices_mlops.modeling.distributions import (
    create_distribution_transformer,
    resolve_distribution_transformations,
)
from house_prices_mlops.modeling.encoding import (
    ORDINAL_ORDERS,
    _available_columns,
    _build_categorical_pipeline,
    _nominal_columns,
    create_onehot_encoder,
    create_ordinal_encoder,
)
from house_prices_mlops.modeling.scaling import create_scaler


def _validate_predictors(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X debe ser un DataFrame de predictores.")
    if {"Id", "SalePrice"}.intersection(X.columns):
        raise ValueError("Excluya Id y SalePrice de X; entregue SalePrice mediante y.")
    return X.copy()

def _numeric_columns(X, *, ordinal_orders):
    return [
        column
        for column in X
        if column not in ordinal_orders and pd.api.types.is_numeric_dtype(X[column])
    ]

def _build_numeric_pipeline(distributions, scaling):
    """Construye imputación por mediana, distribuciones configuradas y escalado opcional."""
    branches = [
        (
            method,
            create_distribution_transformer(method),
            partial(
                _available_columns,
                columns=[var for var, chosen in distributions.items() if chosen == method],
            ),
        )
        for method in dict.fromkeys(distributions.values())
    ]
    distribution_step = ColumnTransformer(
        branches,
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("distributions", distribution_step),
            ("scaler", "passthrough" if scaling is None else create_scaler(scaling)),
        ]
    ).set_output(transform="pandas")

def _build_preprocessor(distributions, orders, scaling):
    """Identifica las columnas numéricas, nominales y ordinales disponibles en cada fold."""
    branches = [
        (
            "numeric",
            _build_numeric_pipeline(distributions, scaling),
            partial(_numeric_columns, ordinal_orders=orders),
        ),
        (
            "nominal",
            _build_categorical_pipeline(create_onehot_encoder()),
            partial(_nominal_columns, ordinal_orders=orders),
        ),
    ]
    # Las ramas ordinales individuales conservan los órdenes aprobados cuando
    # la selección elimina columnas. scikit-learn omite las ramas sin columnas disponibles.
    for column, categories in orders.items():
        branches.append(
            (
                f"ordinal_{column}",
                _build_categorical_pipeline(create_ordinal_encoder({column: categories})),
                partial(_available_columns, columns=[column]),
            )
        )
    return ColumnTransformer(branches, remainder="drop").set_output(transform="pandas")


def create_predictor_validation_step():
    """Valida los predictores sin modificar sus valores ni aprender estadísticas."""
    return FunctionTransformer(_validate_predictors, feature_names_out="one-to-one")


def create_preprocessor(*, distribution_transformations=None, ordinal_orders=None, scaling=None):
    """Construye las ramas numéricas y categóricas completas, sin ajustarlas."""
    distributions = resolve_distribution_transformations(distribution_transformations)
    orders = ORDINAL_ORDERS if ordinal_orders is None else ordinal_orders
    return _build_preprocessor(distributions, orders, scaling)
