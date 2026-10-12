"""Statistical profile built from data and explicit conceptual metadata."""

import numpy as np
import pandas as pd


def build_profile(df, variable_types):

    rows = []

    for col in df.columns:
        s = df[col]

        row = {
            "var": col,
            "data_type": str(s.dtype),
            "variable_type": variable_types.get(col),
            "unique_count": s.nunique(dropna=True),
            "missing_count": s.isna().sum(),
            "missing_pct": s.isna().mean() * 100,
            **dict.fromkeys(
                [
                    "zero_count",
                    "zero_pct",
                    "min",
                    "max",
                    "mean",
                    "median",
                    "std",
                    "skewness",
                    "kurtosis",
                ],
                np.nan,
            ),
        }

        if pd.api.types.is_numeric_dtype(s):
            row.update(
                {
                    "zero_count": (s == 0).sum(),
                    "zero_pct": (s == 0).mean() * 100,
                    "min": s.min(),
                    "max": s.max(),
                    "mean": s.mean(),
                    "median": s.median(),
                    "std": s.std(),
                    "skewness": s.skew(),
                    "kurtosis": s.kurtosis(),
                }
            )

        rows.append(row)

    profile = pd.DataFrame(rows)

    return profile
