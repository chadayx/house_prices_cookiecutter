import contextlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def script():
    return runpy.run_path("scripts/train.py")


def test_arguments_defaults_and_custom(script):
    args = script["parse_args"]([])
    assert vars(args) == {"alpha": 1.0, "tol": 0.0001, "solver": "auto", "seed": 42}
    args = script["parse_args"](
        ["--alpha", "2.5", "--tol", "0.001", "--solver", "lsqr", "--seed", "7"]
    )
    model = script["build_model"](args)
    params = model["estimator"].get_params()
    for key, value in {"alpha": 2.5, "tol": 0.001, "solver": "lsqr", "random_state": 7}.items():
        assert params[key] == value
    assert model["selection"] == "passthrough"
    numeric = model["preprocessing"].transformers[0][1]
    assert numeric["distributions"].transformers == []
    assert numeric["scaler"].__class__.__name__ == "StandardScaler"


@pytest.mark.parametrize(
    "argv",
    [
        ["--alpha", "-1"],
        ["--alpha", "nan"],
        ["--tol", "0"],
        ["--tol", "inf"],
        ["--solver", "invalid"],
        ["--solver", "lbfgs"],
        ["--seed", "-1"],
    ],
)
def test_invalid_arguments(script, argv):
    with pytest.raises(SystemExit) as error:
        script["parse_args"](argv)
    assert error.value.code == 2


def test_seeding_and_reproducible_custom_evaluation(script):
    args = script["parse_args"](["--solver", "lsqr", "--seed", "7"])
    X = pd.DataFrame({"LotArea": np.arange(1, 21) * 10.0, "Neighborhood": ["A", "B"] * 10})
    y = pd.Series(np.arange(20) * 12.0 + 100)
    script["set_seed"](args.seed)
    first_random = np.random.random(3)
    first = script["evaluate_model"](script["build_model"](args), X, y, seed=args.seed)
    script["set_seed"](args.seed)
    np.testing.assert_array_equal(first_random, np.random.random(3))
    second = script["evaluate_model"](script["build_model"](args), X, y, seed=args.seed)
    pd.testing.assert_frame_equal(first[0], second[0])
    assert first[2] == second[2]


def test_logging_artifacts_and_final_model_inside_run(script, monkeypatch):
    args = script["parse_args"]([])
    X = pd.DataFrame({"LotArea": [10.0, 20.0, 30.0, 40.0], "Neighborhood": ["A", "B", "A", "B"]})
    y = pd.Series([100.0, 120.0, 140.0, 160.0])
    folds = pd.DataFrame(
        {"fold": range(1, 6), "mae": [1.0] * 5, "rmse": [2.0] * 5, "r2": [0.9] * 5}
    )
    summary = folds[["mae", "rmse", "r2"]].agg(["mean", "std"])
    metrics = {
        f"{m}_{s}": float(summary.loc[s, m])
        for m in ["mae", "rmse", "r2"]
        for s in ["mean", "std"]
    }
    state = {"active": False, "artifacts": {}, "metric_steps": []}
    mlflow = script["mlflow"]

    @contextlib.contextmanager
    def run(**kwargs):
        state["active"] = True
        yield SimpleNamespace(info=SimpleNamespace(run_id="run", experiment_id="experiment"))
        state["active"] = False

    def log_model(model, **kwargs):
        assert state["active"]
        assert hasattr(model["estimator"], "coef_")
        assert np.isfinite(model.predict(X)).all()
        assert kwargs["serialization_format"] == "cloudpickle"
        assert kwargs["code_paths"]
        assert any(p.startswith("scikit-learn==") for p in kwargs["pip_requirements"])
        return SimpleNamespace(model_uri="models:/model")

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda value: state.update(uri=value))
    monkeypatch.setattr(mlflow, "set_experiment", lambda value: state.update(experiment=value))
    monkeypatch.setattr(mlflow, "start_run", run)
    monkeypatch.setattr(mlflow, "set_tags", lambda value: None)
    monkeypatch.setattr(mlflow, "log_params", lambda value: state.update(params=value))
    monkeypatch.setattr(
        mlflow, "log_metrics", lambda value, **kw: state["metric_steps"].append((value, kw))
    )
    monkeypatch.setattr(
        mlflow,
        "log_artifact",
        lambda value: state["artifacts"].update({Path(value).name: Path(value).read_text()}),
    )
    monkeypatch.setattr(mlflow.sklearn, "log_model", log_model)
    model = script["build_model"](args)
    assert script["log_results"](model, X, y, folds, summary, metrics, args) == (
        "run",
        "experiment",
        "models:/model",
    )
    assert state["experiment"] == "house-prices-ridge-experiments"
    assert state["params"]["seed"] == 42
    assert set(state["artifacts"]) == {"cv_fold_metrics.csv", "cv_summary.json"}
    assert json.loads(state["artifacts"]["cv_summary.json"])["std_ddof"] == 0
    assert len(state["metric_steps"]) == 6
