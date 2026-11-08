import numpy as np
import pandas as pd

from processing.timezone import floor_to_interval
from sentiment.engagement_weighting import weighted_sentiment
from config.settings import CANDLE_INTERVAL


def aggregate_posts_by_interval(posts: pd.DataFrame, interval: str = None) -> pd.DataFrame:
    """Aggregates scored, weighted posts into one row per candle interval."""
    interval = interval or CANDLE_INTERVAL
    if posts.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "sentiment_mean",
                "sentiment_weighted",
                "positive_ratio",
                "negative_ratio",
                "post_count",
                "engagement_total",
            ]
        )

    posts = posts.copy()
    posts["bucket"] = floor_to_interval(posts["created_at_utc"], interval)

    rows = []
    for bucket, group in posts.groupby("bucket"):
        engagement_total = group[["likes", "reposts", "views", "replies"]].fillna(0).sum().sum()
        rows.append(
            {
                "timestamp_utc": bucket,
                "sentiment_mean": group["sentiment_score"].mean(),
                "sentiment_weighted": weighted_sentiment(group),
                "positive_ratio": (group["sentiment_score"] > 0.1).mean(),
                "negative_ratio": (group["sentiment_score"] < -0.1).mean(),
                "post_count": len(group),
                "engagement_total": float(engagement_total),
            }
        )

    return pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)


def aggregate_news_by_interval(news: pd.DataFrame, interval: str = None) -> pd.DataFrame:
    interval = interval or CANDLE_INTERVAL
    if news.empty:
        return pd.DataFrame(columns=["timestamp_utc", "news_sentiment"])

    news = news.copy()
    news["bucket"] = floor_to_interval(news["published_at_utc"], interval)
    agg = news.groupby("bucket")["sentiment_score"].mean().reset_index()
    agg = agg.rename(columns={"bucket": "timestamp_utc", "sentiment_score": "news_sentiment"})
    return agg
