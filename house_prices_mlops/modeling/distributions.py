"""Decisiones ejecutables de distribución provenientes de DP_Transformations."""

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer

DISTRIBUTION_TRANSFORMATIONS = {
    "LotArea": "log",
    "LotFrontage": "sqrt",
    "1stFlrSF": "log",
    "GrLivArea": "log",
    "MasVnrArea": "log1p",
    "BsmtFinSF1": "log1p",
    "WoodDeckSF": "log1p",
    "OpenPorchSF": "log1p",
}
TARGET_TRANSFORMATION = "log"


def create_distribution_transformer(method):
    """Construye una transformación fija y su inversa, conservando los faltantes."""
    functions = {
        "log": (np.log, np.exp),
        "sqrt": (np.sqrt, np.square),
        "log1p": (np.log1p, np.expm1),
    }
    if method not in functions:
        raise ValueError(f"Transformación de distribución desconocida: {method}")
    function, inverse = functions[method]
    return FunctionTransformer(
        func=function, inverse_func=inverse, feature_names_out="one-to-one",
        check_inverse=False,
    )


def create_distribution_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                method,
                create_distribution_transformer(method),
                [var for var, chosen in DISTRIBUTION_TRANSFORMATIONS.items() if chosen == method],
            )
            for method in ("log", "sqrt", "log1p")
        ],
        remainder="drop",
    )


def resolve_distribution_transformations(transformations=None):
    """Copia las distribuciones configuradas y rechaza identificadores o el objetivo."""
    configured = dict(DISTRIBUTION_TRANSFORMATIONS if transformations is None else transformations)
    if {"Id", "SalePrice"}.intersection(configured):
        raise ValueError("Las distribuciones de X no deben incluir Id ni SalePrice.")
    return configured


def create_target_estimator(estimator, log_target):
    """Clona el estimador y transforma opcionalmente solo su objetivo de entrenamiento."""
    result = clone(estimator)
    if log_target:
        result = TransformedTargetRegressor(
            regressor=result,
            transformer=create_distribution_transformer(TARGET_TRANSFORMATION),
        )
    return result

