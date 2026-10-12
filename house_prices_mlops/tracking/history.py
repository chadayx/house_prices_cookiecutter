"""History event registration and deduplication."""

import pandas as pd


def log(history, var, stage, action, method=None, before=None, after=None):
    duplicate = (
        history["var"].eq(var)
        & history["stage"].eq(stage)
        & history["action"].eq(action)
        & history["method"].fillna("").eq(method or "")
        & history["before"].fillna("").eq(before or "")
        & history["after"].fillna("").eq(after or "")
    )
    if duplicate.any():
        return history, None

    event_id = 1 if history.empty else int(history["id"].max()) + 1
    event = pd.DataFrame(
        [
            {
                "id": event_id,
                "var": var,
                "stage": stage,
                "action": action,
                "method": method,
                "before": before,
                "after": after,
            }
        ]
    )
    history = pd.concat([history, event], ignore_index=True)
    return history, event_id
