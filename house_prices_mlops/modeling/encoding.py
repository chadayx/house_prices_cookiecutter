"""Decisiones explícitas de codificación de 07_encoding.ipynb; no ejecuta el tracker."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

ORDINAL_ORDERS = {'LotShape': ['IR3', 'IR2', 'IR1', 'Reg'],
 'Utilities': ['NoSeWa', 'AllPub'],
 'LandSlope': ['Sev', 'Mod', 'Gtl'],
 'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
 'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
 'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
 'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
 'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
 'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
 'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
 'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
 'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
 'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
 'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
 'GarageFinish': ['None', 'Unf', 'RFn', 'Fin'],
 'GarageQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
 'GarageCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
 'PavedDrive': ['N', 'P', 'Y'],
 'PoolQC': ['None', 'Fa', 'TA', 'Gd', 'Ex']}


def create_ordinal_encoder(orders):
    """Conserva el orden aprobado: desconocidos=-1, faltantes=-2 y None estructural intacto.

    Agrega np.nan únicamente para manejar faltantes, sin alterar el orden ni los
    códigos de las categorías aprobadas. Las columnas deben seguir el mismo
    orden que las claves de orders."""
    return OrdinalEncoder(
        categories=[[*categories, np.nan] for categories in orders.values()],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
    )


def create_onehot_encoder():
    """Aprende categorías durante fit; las desconocidas generan indicadores en cero.

    np.nan es una categoría diferenciada si aparece durante el ajuste. Un faltante
    no observado en entrenamiento sigue la política de ignorar categorías
    desconocidas. No imputa estadísticas ni sustituye ausencias estructurales."""
    return OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False)


def normalize_encoding_missing(X):
    """Normaliza los marcadores faltantes de pandas y None a np.nan, sin imputarlos."""
    result = X.astype(object).copy()
    return result.where(result.notna(), np.nan)


def _available_columns(X, *, columns):
    return [column for column in columns if column in X.columns]

def _nominal_columns(X, *, ordinal_orders):
    return [
        column
        for column in X
        if column not in ordinal_orders and not pd.api.types.is_numeric_dtype(X[column])
    ]

def _build_categorical_pipeline(encoder):
    """Normaliza marcadores faltantes sin imputar y aplica el encoder seleccionado."""
    return Pipeline(
        [
            (
                "missing_markers",
                FunctionTransformer(
                    normalize_encoding_missing,
                    feature_names_out="one-to-one",
                ),
            ),
            ("encoder", encoder),
        ]
    )

def create_encoding_preprocessor(*, nominal_columns, ordinal_columns, ordinal_orders=None):
    """Construye la codificación para las columnas disponibles tras la selección.

    Recibe solo predictores, sin Id ni SalePrice. Las columnas restantes pasan
    sin cambios; quien llama debe indicar todas las categorías que desea codificar.
    OverallQual y OverallCond conservan su representación numérica.
    Este componente no es un pipeline completo de modelado ni de imputación."""
    nominal_columns = list(nominal_columns)
    ordinal_columns = list(ordinal_columns)
    orders = ORDINAL_ORDERS if ordinal_orders is None else ordinal_orders
    all_columns = nominal_columns + ordinal_columns
    if len(all_columns) != len(set(all_columns)):
        raise ValueError("Las columnas de encoding deben ser únicas y no solaparse.")
    if {"Id", "SalePrice", "OverallQual", "OverallCond"}.intersection(all_columns):
        raise ValueError("No codifique Id, SalePrice, OverallQual ni OverallCond.")
    missing_orders = set(ordinal_columns) - set(orders)
    if missing_orders:
        raise ValueError(f"Faltan órdenes aprobados para: {sorted(missing_orders)}")

    branches = []
    if nominal_columns:
        branches.append(
            ("onehot", _build_categorical_pipeline(create_onehot_encoder()), nominal_columns)
        )
    if ordinal_columns:
        chosen_orders = {column: orders[column] for column in ordinal_columns}
        branches.append(
            (
                "ordinal",
                _build_categorical_pipeline(create_ordinal_encoder(chosen_orders)),
                ordinal_columns,
            )
        )
    return ColumnTransformer(branches, remainder="passthrough").set_output(transform="pandas")
