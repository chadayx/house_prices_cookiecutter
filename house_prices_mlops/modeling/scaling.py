"""Alternativas de escalado dependientes del modelo investigadas en 08_scaling.ipynb."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def create_scaler(method, **params):
    """Construye un escalador sin ajustar; sus estadísticas se aprenden en cada fold.

    No selecciona un método por defecto, no imputa faltantes ni lee planes del
    tracker. Conserva copy=True, el comportamiento predeterminado de scikit-learn."""
    scalers = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
    }
    if method not in scalers:
        raise ValueError(f"Método de scaling desconocido: {method}")
    if params.get("copy") is False:
        raise ValueError("Use copy=True para conservar los datos originales.")
    return scalers[method](**params)


def create_scaling_preprocessor(*, columns, method, **params):
    """Escala los predictores seleccionados explícitamente y conserva las otras columnas.

    Configura las columnas después de selección o codificación, según corresponda.
    Es únicamente un componente de escalado, no el pipeline experimental completo."""
    columns = list(columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Indique una lista no vacía de columnas únicas para scaling.")
    if {"Id", "SalePrice"}.intersection(columns):
        raise ValueError("Id y SalePrice no deben escalarse como predictores.")
    return ColumnTransformer(
        [("scaling", create_scaler(method, **params), columns)],
        remainder="passthrough",
    ).set_output(transform="pandas")
