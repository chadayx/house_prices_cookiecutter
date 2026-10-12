"""Plan registration with stable identifiers and deduplication."""

import pandas as pd


def add_plan(plans, var, stage, action, method, params=None):
    if params is None:
        params = {}

    # Evitar duplicados
    duplicate = (
        plans["var"].eq(var)
        & plans["stage"].eq(stage)
        & plans["action"].eq(action)
        & plans["method"].eq(method)
    )

    if duplicate.any():
        return plans

    new_plan = {
        "id": 1 if plans.empty else int(plans["id"].max()) + 1,
        "var": var,
        "stage": stage,
        "action": action,
        "method": method,
        "params": params,
        "status": "planned",
    }

    plans = pd.concat([plans, pd.DataFrame([new_plan])], ignore_index=True)
    return plans
