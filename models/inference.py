import pandas as pd

from config.settings import HORIZONS
from models.multi_horizon import predict_multi_horizon


def generate_prediction_matrix(models: dict, latest_row: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Builds the n+0..n+5 prediction matrix for the most recent available data point."""
    preds = predict_multi_horizon(models, latest_row, feature_cols)
    reference_price = preds["n0"].iloc[0]

    rows = []
    for h in HORIZONS:
        predicted = preds[f"n{h}"].iloc[0] if h != 0 else reference_price
        change_pct = (predicted - reference_price) / reference_price * 100
        direction = "flat"
        if h != 0:
            direction = "up" if change_pct > 0 else "down" if change_pct < 0 else "flat"
        rows.append(
            {
                "horizon": f"n+{h}",
                "predicted_price": predicted,
                "change_pct": change_pct,
                "direction": direction,
            }
        )
    return pd.DataFrame(rows)
