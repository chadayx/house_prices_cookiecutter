"""Reconstrucción en memoria de estados de preparación."""

from house_prices_mlops.dataset import (
    get_test_data,
    get_training_data,
    load_split_manifest,
)

from house_prices_mlops.preparation.structure import prepare_structure
from house_prices_mlops.preparation.missing import prepare_missing
from house_prices_mlops.preparation.feature_engineering import prepare_features


PREPARATION_STEPS = (
    ("structure", prepare_structure),
    ("missing", prepare_missing),
    ("common", prepare_features),
)


def load_stage(stage: str, *, split: str = "train"):
    """Recupera una partición y aplica las etapas solicitadas."""

    if stage == "raw":
        data, _ = load_split_manifest()
        return data.copy()

    if stage == "train":
        return get_training_data()

    if stage == "test":
        return get_test_data()

    if stage not in dict(PREPARATION_STEPS):
        raise ValueError(f"Etapa desconocida: {stage}")

    if split not in ("train", "test"):
        raise ValueError(f"Partición desconocida: {split}")

    data = (
        get_training_data()
        if split == "train"
        else get_test_data()
    ).copy()

    for name, transform in PREPARATION_STEPS:
        data = transform(data)

        if name == stage:
            return data