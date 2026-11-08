import numpy as np
import pandas as pd

from config.settings import HORIZONS


def train_multi_horizon(train_df: pd.DataFrame, feature_cols: list[str], model_builder) -> dict:
    """Trains one model per forecast horizon (n+1..n+5; n+0 is the reference price, not fitted)."""
    models = {}
    for h in HORIZONS:
        if h == 0:
            continue
        target_col = f"target_n{h}"
        model = model_builder()
        model.fit(train_df[feature_cols], train_df[target_col])
        models[h] = model
    return models


def predict_multi_horizon(models: dict, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Returns a dataframe with one predicted_n{h} column per horizon, aligned to df's index."""
    preds = pd.DataFrame(index=df.index)
    preds["n0"] = df["price"].values
    for h, model in models.items():
        preds[f"n{h}"] = model.predict(df[feature_cols])
    return preds
