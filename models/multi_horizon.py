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
    """Returns a dataframe with one predicted log-return column (n{h}) per horizon
    (models are fit on log-return targets -- see features/feature_pipeline.py:add_targets),
    plus n0 = current price for reference. Use predict_multi_horizon_price() instead
    whenever a price-level prediction is actually needed (metrics, strategy returns,
    the prediction matrix) -- this function's n{h} values are NOT prices."""
    preds = pd.DataFrame(index=df.index)
    preds["n0"] = df["price"].values
    for h, model in models.items():
        preds[f"n{h}"] = model.predict(df[feature_cols])
    return preds


def predict_multi_horizon_price(models: dict, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Same as predict_multi_horizon, but converts each horizon's predicted log-return
    back into a predicted price level: price * exp(predicted_log_return). This is what
    callers that need a dollar-denominated prediction (the forecast matrix, MAE/RMSE,
    the trading strategy) should use."""
    current_price = df["price"].values
    preds = pd.DataFrame(index=df.index)
    preds["n0"] = current_price
    for h, model in models.items():
        predicted_log_return = model.predict(df[feature_cols])
        preds[f"n{h}"] = current_price * np.exp(predicted_log_return)
    return preds
