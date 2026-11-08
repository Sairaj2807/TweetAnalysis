import numpy as np
import pandas as pd

from config.settings import HORIZONS
from features.price_features import add_price_features
from features.sentiment_features import add_sentiment_features

PRICE_ONLY_FEATURES = ["price", "return_1", "return_3", "return_5", "rolling_volatility", "price_momentum"]
SENTIMENT_FEATURES = [
    "sentiment_mean",
    "sentiment_weighted",
    "positive_ratio",
    "negative_ratio",
    "post_count",
    "engagement_total",
    "sentiment_change",
    "sentiment_lag_1",
]
NEWS_FEATURES = ["news_sentiment"]

FEATURE_SETS = {
    "price_only": PRICE_ONLY_FEATURES,
    "price_plus_sentiment": PRICE_ONLY_FEATURES + SENTIMENT_FEATURES,
    "price_plus_sentiment_plus_news": PRICE_ONLY_FEATURES + SENTIMENT_FEATURES + NEWS_FEATURES,
}


def build_features(merged: pd.DataFrame) -> pd.DataFrame:
    """merged must have columns: timestamp_utc, price, sentiment_mean, sentiment_weighted,
    positive_ratio, negative_ratio, post_count, engagement_total, news_sentiment (optional)."""
    df = merged.sort_values("timestamp_utc").reset_index(drop=True)
    df = add_price_features(df, price_col="price")

    for col in ["sentiment_mean", "sentiment_weighted", "positive_ratio", "negative_ratio", "post_count", "engagement_total"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    df = add_sentiment_features(df, sentiment_col="sentiment_mean")

    if "news_sentiment" not in df.columns:
        df["news_sentiment"] = 0.0
    df["news_sentiment"] = df["news_sentiment"].fillna(0.0)

    for col in SENTIMENT_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Adds target_n0..target_nK log-return-based price targets for each horizon."""
    df = df.copy()
    for h in HORIZONS:
        if h == 0:
            df[f"target_n{h}"] = df["price"]
        else:
            df[f"target_n{h}"] = df["price"].shift(-h)
    return df


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows with NaN features/targets introduced by rolling windows and shifts."""
    required_cols = FEATURE_SETS["price_plus_sentiment_plus_news"] + [f"target_n{h}" for h in HORIZONS]
    return df.dropna(subset=required_cols).reset_index(drop=True)
