import numpy as np
import pandas as pd


def add_price_features(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    df = df.copy()
    log_price = np.log(df[price_col])

    df["return_1"] = log_price.diff(1)
    df["return_3"] = log_price.diff(3)
    df["return_5"] = log_price.diff(5)
    df["rolling_volatility"] = df["return_1"].rolling(window=12, min_periods=3).std()
    df["price_momentum"] = df[price_col] / df[price_col].shift(5) - 1
    return df
