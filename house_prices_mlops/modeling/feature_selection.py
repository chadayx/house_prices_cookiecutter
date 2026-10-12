"""Análisis de baja variabilidad con entrenamiento y aplicación explícita del resultado.

Aprende la lista de eliminaciones en cada fold de entrenamiento y aplica esa
misma lista a entrenamiento y validación. No la aprende antes de validación cruzada.
No lee planes ni considera definitiva una lista fija de columnas."""



import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def correlation_ratio(categories, values):
    """Calcula la asociación categórica eta, incluyendo los faltantes como un grupo."""
    codes, _ = pd.factorize(categories)
    mean = values.mean()
    denominator = ((values - mean) ** 2).sum()
    if denominator == 0:
        return np.nan
    numerator = 0.0
    for code in np.unique(codes):
        group = values.iloc[np.flatnonzero(codes == code)]
        numerator += len(group) * (group.mean() - mean) ** 2
    return float(np.sqrt(np.clip(numerator / denominator, 0, 1)))


def analyze_low_variability(X, y, *, min_dominant_pct=95):
    """Compara la dominancia con |Pearson| para numéricas y eta para categóricas.

    X debe contener solo predictores. Los faltantes cuentan para la dominancia,
    como en el notebook. La asociación numérica utiliza pares observados; si un
    predictor numérico está totalmente faltante, su asociación es desconocida
    y no se elimina automáticamente."""
    if not 0 <= min_dominant_pct <= 100:
        raise ValueError("min_dominant_pct debe estar entre 0 y 100.")
    if X.empty or not X.columns.is_unique:
        raise ValueError("X debe contener filas y columnas únicas.")
    if {"Id", "SalePrice"}.intersection(X.columns):
        raise ValueError("Excluya Id y SalePrice de X; entregue el target mediante y.")
    if isinstance(y, pd.Series):
        if not y.index.equals(X.index):
            raise ValueError("Los índices de X e y deben coincidir en el mismo orden.")
    else:
        y = pd.Series(y, index=X.index)
    if not pd.api.types.is_numeric_dtype(y) or not np.isfinite(y.to_numpy()).all():
        raise ValueError("y debe ser numérico, sin nulos ni valores infinitos.")
    if y.nunique() < 2:
        raise ValueError("y debe variar para evaluar su asociación con los predictores.")

    rows = []
    for var in X.columns:
        feature = X[var]
        counts = feature.value_counts(normalize=True, dropna=False)
        numeric = pd.api.types.is_numeric_dtype(feature)
        if numeric:
            observed = feature.notna()
            if not observed.any():
                association = np.nan
            elif feature.loc[observed].nunique() == 1:
                association = 0.0
            else:
                association = abs(feature.corr(y))
        else:
            association = correlation_ratio(feature, y)
        dominant_pct = float(counts.iloc[0] * 100)
        rows.append({
            "var": var,
            "unique_count": int(feature.nunique(dropna=False)),
            "dominant_value": counts.index[0],
            "dominant_pct": dominant_pct,
            "low_variability": dominant_pct >= min_dominant_pct,
            "association_method": "pearson_abs" if numeric else "eta",
            "target_association": association,
        })
    return pd.DataFrame(rows)


def fit_low_variability_selection(X, y, *, max_target_association, min_dominant_pct=95):
    """Aprende eliminaciones combinando dominancia y un límite explícito de asociación.

    El notebook definió 95 % de dominancia, pero no un límite de asociación.
    Por ello max_target_association es obligatorio; no se inventa un valor
    predeterminado para los experimentos."""
    if not 0 <= max_target_association <= 1:
        raise ValueError("max_target_association debe estar entre 0 y 1.")
    review = analyze_low_variability(X, y, min_dominant_pct=min_dominant_pct)
    return review.loc[
        review["low_variability"] & review["target_association"].le(max_target_association),
        "var",
    ].tolist()


def apply_feature_selection(X, *, drop_columns):
    """Aplica la lista del fold de entrenamiento sobre una copia, sin aprender parámetros."""
    return X.drop(columns=list(drop_columns)).copy()


class LowVariabilitySelector(TransformerMixin, BaseEstimator):
    """Aprende eliminaciones durante fit y reutiliza la misma selección en transform."""

    def __init__(self, *, max_target_association, min_dominant_pct=95):
        self.max_target_association = max_target_association
        self.min_dominant_pct = min_dominant_pct

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un DataFrame.")
        drops = fit_low_variability_selection(
            X, y,
            max_target_association=self.max_target_association,
            min_dominant_pct=self.min_dominant_pct,
        )
        self.drop_columns_ = drops
        self.columns_in_ = X.columns.copy()
        self.feature_names_in_ = X.columns.to_numpy(copy=True)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ["drop_columns_", "columns_in_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un DataFrame.")
        if not X.columns.equals(self.columns_in_):
            raise ValueError("Las columnas y su orden deben coincidir con las usadas en fit().")
        return apply_feature_selection(X, drop_columns=self.drop_columns_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["drop_columns_", "columns_in_"])
        if input_features is not None and not np.array_equal(input_features, self.feature_names_in_):
            raise ValueError("Los nombres deben coincidir con las columnas usadas en fit().")
        return self.columns_in_[~self.columns_in_.isin(self.drop_columns_)].to_numpy(copy=True)


def create_selection_step(*, use_selection=False, max_target_association=None, min_dominant_pct=95):
    """Construye la selección opcional por fold sin aprender parámetros todavía."""
    if not use_selection:
        return "passthrough"
    if max_target_association is None:
        raise ValueError("Indique max_target_association para activar selección.")
    return LowVariabilitySelector(
        max_target_association=max_target_association,
        min_dominant_pct=min_dominant_pct,
    )

