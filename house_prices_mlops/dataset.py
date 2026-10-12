"""Stable train/holdout partition of the original House Prices data."""

import hashlib
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import typer

from house_prices_mlops.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()
DEFAULT_INPUT = RAW_DATA_DIR / "train.csv"
DEFAULT_MANIFEST = INTERIM_DATA_DIR / "split_manifest.csv"


def load_split_manifest(input_path: Path = DEFAULT_INPUT, manifest_path: Path = DEFAULT_MANIFEST):

    data = pd.read_csv(input_path)
    if not {"Id", "SalePrice"}.issubset(data.columns):
        raise ValueError("El dataset debe contener Id y SalePrice.")
    if data["Id"].isna().any() or data["Id"].duplicated().any():
        raise ValueError("Id no debe contener nulos ni duplicados.")
    if len(data) < 2:
        raise ValueError("Se necesitan al menos dos viviendas.")
    
    data = data.sort_values("Id").reset_index(drop=True)
    schema = "\n".join(f"{col}:{dtype}" for col, dtype in data.dtypes.items())
    fingerprint = hashlib.sha256(
        (schema + "\n" + data.to_csv(index=False, lineterminator="\n")).encode("utf-8")
    ).hexdigest()
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        required = {"Id", "split", "dataset_sha256", "random_state", "test_size"}
        valid = required.issubset(manifest.columns)
        if valid:
            valid = (
                len(manifest) == len(data)
                and not manifest["Id"].isna().any()
                and not manifest["Id"].duplicated().any()
                and set(manifest["Id"]) == set(data["Id"])
                and manifest["dataset_sha256"].eq(fingerprint).all()
                and manifest["random_state"].eq(42).all()
                and manifest["test_size"].eq(0.2).all()
                and manifest["split"].isin(["train", "test"]).all()
                and manifest["split"].eq("test").sum() == math.ceil(len(data) * 0.2)
            )
        if not valid:
            raise ValueError(
                "El manifiesto es inválido o los datos cambiaron. No se sobrescribió. "
                "Revise los datos y gestione una nueva versión explícitamente."
            )
    else:
        _, test_ids = train_test_split(data["Id"], test_size=0.2, random_state=42)
        manifest = pd.DataFrame({"Id": data["Id"]})
        manifest["split"] = "train"
        manifest.loc[manifest["Id"].isin(test_ids), "split"] = "test"
        manifest["dataset_sha256"] = fingerprint
        manifest["random_state"] = 42
        manifest["test_size"] = 0.2
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x", encoding="utf-8", newline="") as handle:
            manifest.to_csv(handle, index=False)
    return data, manifest


def get_training_data(
    input_path: Path = DEFAULT_INPUT, manifest_path: Path = DEFAULT_MANIFEST
) -> pd.DataFrame:
    """Return original training rows, including Id and SalePrice."""
    data, manifest = load_split_manifest(input_path, manifest_path)
    ids = manifest.loc[manifest["split"] == "train", "Id"]
    return data.loc[data["Id"].isin(ids)].copy()


def get_test_data(
    input_path: Path = DEFAULT_INPUT, manifest_path: Path = DEFAULT_MANIFEST
) -> pd.DataFrame:
    """Return reserved holdout rows, including Id and SalePrice."""
    data, manifest = load_split_manifest(input_path, manifest_path)
    ids = manifest.loc[manifest["split"] == "test", "Id"]
    return data.loc[data["Id"].isin(ids)].copy()


def get_predictors_and_target(data: pd.DataFrame):
    """Exclude Id and SalePrice from model predictors."""
    return data.drop(columns=["Id", "SalePrice"]).copy(), data["SalePrice"].copy()


@app.command()
def main(input_path: Path = DEFAULT_INPUT, manifest_path: Path = DEFAULT_MANIFEST):
    """Validate or create the partition before preparing data."""
    _, manifest = load_split_manifest(input_path, manifest_path)
    typer.echo(f"Partición validada: {manifest['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    app()
