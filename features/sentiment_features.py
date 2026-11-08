import pandas as pd


def add_sentiment_features(df: pd.DataFrame, sentiment_col: str = "sentiment_mean") -> pd.DataFrame:
    df = df.copy()
    df["sentiment_change"] = df[sentiment_col].diff(1)
    df["sentiment_lag_1"] = df[sentiment_col].shift(1)
    return df
