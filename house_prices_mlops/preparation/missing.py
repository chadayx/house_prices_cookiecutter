import pandas as pd


def normalize_structural_absence(data: pd.DataFrame) -> pd.DataFrame:
    """Representa las ausencias estructurales mediante 'None'."""

    structural_cat = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "BsmtQual",
        "BsmtCond",
    ]

    result = data.copy()
    result[structural_cat] = result[structural_cat].fillna("None")
    return result


def normalize_masonry_absence(data: pd.DataFrame) -> pd.DataFrame:
    """Representa MasVnrType faltante como ausencia cuando MasVnrArea es cero."""
    result = data.copy()
    mask_no_veneer = result["MasVnrType"].isna() & result["MasVnrArea"].eq(0)
    result.loc[mask_no_veneer, "MasVnrType"] = "None"
    return result


def prepare_missing(data: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta las transformaciones aprobadas de Missing Values."""

    result = normalize_structural_absence(data)
    result = normalize_masonry_absence(result)

    return result
