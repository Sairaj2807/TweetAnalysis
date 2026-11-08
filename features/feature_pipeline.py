import numpy as np
import pandas as pd

from config.settings import HORIZONS
from features.price_features import add_price_features
from features.sentiment_features import add_sentiment_features

PRICE_ONLY_FEATURES = ["return_1", "return_3", "return_5", "rolling_volatility", "price_momentum"]
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

# engagement_total and post_count aren't comparable across the two historical tweet
# sources combined in run_historical_backtest_combined.py: the 2021-23 Kaggle source
# has engagement_total == 0 on every row (no per-tweet engagement data), while the
# 2014-19 tweets.csv source has it > 0 on every row (it does have that data) -- so
# the column is a near-perfect indicator of which period a row comes from, not a
# genuine sentiment signal. Use this trimmed list for the combined backtest only.
SENTIMENT_FEATURES_NO_VOLUME = [c for c in SENTIMENT_FEATURES if c not in ("engagement_total", "post_count")]

# Historical backtests have no real news data at all -- Tavily can only be queried
# for current news, not "as of" a past date, so news_sentiment is a constant 0
# throughout both historical backtests. Including "price_plus_sentiment_plus_news"
# there would just duplicate "price_plus_sentiment" numbers (or differ only by
# incidental model randomness), not reflect an actual third experiment -- see README.
HISTORICAL_FEATURE_SETS = {
    "price_only": PRICE_ONLY_FEATURES,
    "price_plus_sentiment": PRICE_ONLY_FEATURES + SENTIMENT_FEATURES,
}

COMBINED_HISTORICAL_FEATURE_SETS = {
    "price_only": PRICE_ONLY_FEATURES,
    "price_plus_sentiment": PRICE_ONLY_FEATURES + SENTIMENT_FEATURES_NO_VOLUME,
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
    """Adds target_n0..target_nK (log-return targets the models actually fit on) and
    future_price_n0..future_price_nK (the true future price level, for metrics/strategy
    evaluation which need dollar/price terms, not log-returns).

    Targets are log-returns -- log(future_price / current_price) -- not raw future
    price, because fitting tree models directly on absolute price level (with price
    itself as a feature) means the model can only split on price ranges seen in
    training; on a trending asset like BTC, out-of-sample prices routinely fall
    outside that range and the model can't extrapolate past it. Log-returns are
    scale-invariant, so the model generalizes across price regimes instead of
    memorizing one.
    """
    df = df.copy()
    for h in HORIZONS:
        if h == 0:
            df[f"target_n{h}"] = df["price"]
            df[f"future_price_n{h}"] = df["price"]
        else:
            future_price = df["price"].shift(-h)
            df[f"future_price_n{h}"] = future_price
            df[f"target_n{h}"] = np.log(future_price / df["price"])
    return df


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows with NaN features/targets introduced by rolling windows and shifts."""
    required_cols = (
        FEATURE_SETS["price_plus_sentiment_plus_news"]
        + [f"target_n{h}" for h in HORIZONS]
        + [f"future_price_n{h}" for h in HORIZONS]
    )
    return df.dropna(subset=required_cols).reset_index(drop=True)
