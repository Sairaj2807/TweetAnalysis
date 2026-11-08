import numpy as np
import pandas as pd


def persistence_predict(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Baseline: predicted price at n+h is just the current price (no change)."""
    return df["price"].values
