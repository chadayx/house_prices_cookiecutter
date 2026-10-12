"""Entrenamiento Ridge, validación cruzada, ajuste final y MLflow."""

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import sys
import tempfile

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from house_prices_mlops.dataset import get_predictors_and_target
from house_prices_mlops.modeling.encoding import ORDINAL_ORDERS
from house_prices_mlops.modeling.pipelines import build_model_pipeline
from house_prices_mlops.pipeline import load_stage


def parse_args(argv=None):
    """Expone los cuatro argumentos y valida la configuración solicitada."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=0.0001)
    parser.add_argument(
        "--solver",
        type=str,
        default="auto",
        choices=("auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"),
        help="Solver de Ridge; lbfgs requiere positive=True y no pertenece a este experimento.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not math.isfinite(args.alpha) or args.alpha < 0:
        parser.error("alpha debe ser finito y mayor o igual a cero.")
    if not math.isfinite(args.tol) or args.tol <= 0:
        parser.error("tol debe ser finito y mayor que cero.")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed debe estar entre 0 y 4294967295.")
    return args


def set_seed(seed):
    """Fija las semillas globales; Ridge y KFold reciben también la misma semilla."""
    np.random.seed(seed)
    random.seed(seed)


def load_training_data():
    return get_predictors_and_target(load_stage("common"))


def build_model(args=None):
    """Construye el pipeline base sin ajustar ni duplicar transformaciones."""
    args = parse_args([]) if args is None else args
    estimator = Ridge(alpha=args.alpha, tol=args.tol, solver=args.solver, random_state=args.seed)
    return build_model_pipeline(
        estimator,
        use_selection=False,
        distribution_transformations={},
        scaling="standard",
        log_target=False,
    )


def evaluate_model(model, X, y, seed=42):
    """Evalúa cinco folds y devuelve métricas por fold"""
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, error_score="raise")
    folds = pd.DataFrame({
            "fold": np.arange(1, 6),
            "mae": -results["test_mae"],
            "rmse": -results["test_rmse"],
            "r2": results["test_r2"]}
    )
    if not np.isfinite(folds[["mae", "rmse", "r2"]].to_numpy()).all():
        raise ValueError("La validación produjo métricas no finitas.")
    summary = folds[["mae", "rmse", "r2"]].agg(["mean", "std"])
    summary.loc["std"] = folds[["mae", "rmse", "r2"]].std(ddof=0)
    metrics = {
        f"{metric}_{statistic}": float(summary.loc[statistic, metric])
        for metric in ("mae", "rmse", "r2")
        for statistic in ("mean", "std")
    }
    return folds, summary, metrics


def train_final_model(model, X, y):
    """Ajusta todo el pipeline con entrenamiento; el holdout permanece reservado."""
    return model.fit(X, y)


def runtime_versions():
    """Documenta las versiones reales sin modificar el entorno ni sus dependencias."""
    packages = ("mlflow", "scikit-learn", "numpy", "pandas", "scipy", "cloudpickle", "joblib")
    return {"python": platform.python_version(), **{name: version(name) for name in packages}}


def log_results(model, X, y, folds, summary, metrics, args):
    """Registra evaluación, artefactos y el pipeline ajustado dentro de la misma corrida."""
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("house-prices-ridge-cookiecutter")
    versions = runtime_versions()
    configuration = {
        "alpha": args.alpha,
        "tol": args.tol,
        "solver": args.solver,
        "seed": args.seed,
        "model": "Ridge",
        "data_stage": "common",
        "split": "train",
        "training_rows": len(X),
        "predictors_before_encoding": X.shape[1],
        "use_selection": False,
        "distribution_transformations": "{}",
        "scaling": "standard",
        "log_target": False,
        "numeric_imputation": "median",
        "keep_empty_features": True,
        "cv": "KFold",
        "cv_n_splits": 5,
        "cv_shuffle": True,
        "cv_random_state": args.seed,
        "metric_std_ddof": 0,
        "ordinal_orders": json.dumps(ORDINAL_ORDERS, ensure_ascii=False),
        "onehot_drop": "None",
        "onehot_handle_unknown": "ignore",
        "ordinal_unknown_value": -1,
        "ordinal_missing_value": -2,
        **{f"ridge_{key}": value for key, value in model["estimator"].get_params().items()},
        **{f"version_{key}": value for key, value in versions.items()},
    }
    with mlflow.start_run(run_name="ridge_baseline_standard_no_selection_no_distributions") as run:
        mlflow.log_params(configuration)
        mlflow.set_tags({"phase": "baseline", "evaluation": "5-fold CV on training only"})
        mlflow.log_metrics(metrics)
        for row in folds.itertuples(index=False):
            mlflow.log_metrics(
                {"fold_mae": row.mae, "fold_rmse": row.rmse, "fold_r2": row.r2}, step=int(row.fold)
            )
        with tempfile.TemporaryDirectory() as directory:
            fold_path = Path(directory) / "cv_fold_metrics.csv"
            summary_path = Path(directory) / "cv_summary.json"
            folds.to_csv(fold_path, index=False)
            summary_path.write_text(
                json.dumps(
                    {
                        "mean": summary.loc["mean"].to_dict(),
                        "std": summary.loc["std"].to_dict(),
                        "std_ddof": 0,
                        "versions": versions,
                    },
                    ensure_ascii=False,
                    indent=4,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            mlflow.log_artifact(str(fold_path))
            mlflow.log_artifact(str(summary_path))
        train_final_model(model, X, y)
        signature = infer_signature(X, model.predict(X))
        # Incluye el código requerido por las funciones del pipeline y fija las versiones.
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            serialization_format="cloudpickle",
            code_paths=[str(PROJECT_ROOT / "house_prices_mlops")],
            pip_requirements=[
                f"{name}=={value}" for name, value in versions.items() if name != "python"
            ],
        )
        return run.info.run_id, run.info.experiment_id, model_info.model_uri


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    X, y = load_training_data()
    model = build_model(args)
    print(f"Entrenamiento: {len(X)} viviendas, {X.shape[1]} predictores", flush=True)
    folds, summary, metrics = evaluate_model(model, X, y, seed=args.seed)
    print(folds.to_string(index=False))
    print("\nPromedios y desviaciones estándar (ddof=0):")
    print(summary.to_string(), flush=True)
    run_id, experiment_id, model_uri = log_results(model, X, y, folds, summary, metrics, args)
    print(f"MLflow run: {run_id}")
    print(f"MLflow model: {model_uri}")
    print(f"MLflow URL: http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}")


if __name__ == "__main__":
    main()
