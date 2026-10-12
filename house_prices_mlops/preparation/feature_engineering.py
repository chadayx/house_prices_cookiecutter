"""Funciones de ingeniería de variables trasladadas del notebook original."""

import numpy as np
import pandas as pd


def create_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["HouseAge"] = result["YrSold"] - result["YearBuilt"]
    result["RemodAge"] = result["YrSold"] - result["YearRemodAdd"]
    result["GarageAge"] = np.where(
        result["GarageYrBlt"].eq(0), 0, result["YrSold"] - result["GarageYrBlt"]
    )
    return result


def normalize_remodel_age(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["RemodAge"] = result["RemodAge"].clip(lower=0)
    return result


def create_aggregate_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["TotalSF"] = result["TotalBsmtSF"] + result["GrLivArea"]
    result["TotalBathrooms"] = (
        result["FullBath"] + 0.5 * result["HalfBath"]
        + result["BsmtFullBath"] + 0.5 * result["BsmtHalfBath"]
    )
    result["TotalPorchSF"] = (
        result["WoodDeckSF"] + result["OpenPorchSF"] + result["EnclosedPorch"]
        + result["3SsnPorch"] + result["ScreenPorch"]
    )
    return result


def create_presence_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["HasGarage"] = result["GarageArea"].gt(0).astype(int)
    result["HasBasement"] = result["TotalBsmtSF"].gt(0).astype(int)
    result["HasPool"] = result["PoolArea"].gt(0).astype(int)
    result["HasFireplace"] = result["Fireplaces"].gt(0).astype(int)
    result["IsRemodeled"] = result["YearBuilt"].ne(result["YearRemodAdd"]).astype(int)
    return result


def create_quality_area_interaction(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["QualGrLivArea"] = result["OverallQual"] * result["GrLivArea"]
    return result


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    result = create_temporal_features(data)
    result = normalize_remodel_age(result)
    result = create_aggregate_features(result)
    result = create_presence_features(result)
    return create_quality_area_interaction(result)
