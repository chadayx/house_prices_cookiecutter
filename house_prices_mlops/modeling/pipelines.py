"""Orquestación de los componentes de selección, preprocesamiento y estimador."""

from sklearn.pipeline import Pipeline

# Las reexportaciones conservan los imports anteriores sin duplicar constructores.
from house_prices_mlops.modeling.distributions import (
    create_distribution_preprocessor as create_distribution_preprocessor,
)
from house_prices_mlops.modeling.distributions import (
    create_target_estimator,
)
from house_prices_mlops.modeling.encoding import (
    create_encoding_preprocessor as create_encoding_preprocessor,
)
from house_prices_mlops.modeling.feature_selection import create_selection_step
from house_prices_mlops.modeling.preprocessing import (
    create_predictor_validation_step,
    create_preprocessor,
)
from house_prices_mlops.modeling.scaling import (
    create_scaling_preprocessor as create_scaling_preprocessor,
)


def build_model_pipeline(
    estimator,
    *,
    use_selection=False,
    max_target_association=None,
    min_dominant_pct=95,
    scaling=None,
    log_target=False,
    distribution_transformations=None,
    ordinal_orders=None,
):
    """Construye un experimento sin ajustar; la selección recibe el objetivo original."""
    selection = create_selection_step(
        use_selection=use_selection,
        max_target_association=max_target_association,
        min_dominant_pct=min_dominant_pct,
    )
    preprocessor = create_preprocessor(
        distribution_transformations=distribution_transformations,
        ordinal_orders=ordinal_orders,
        scaling=scaling,
    )
    return Pipeline([
        ("predictors", create_predictor_validation_step()),
        ("selection", selection),
        ("preprocessing", preprocessor),
        ("estimator", create_target_estimator(estimator, log_target)),
    ])
