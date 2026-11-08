import numpy as np
import pandas as pd

MAX_WEIGHT_PERCENTILE = 0.99  # cap extreme weights so a few large accounts don't dominate


def add_engagement_weight(posts: pd.DataFrame) -> pd.DataFrame:
    posts = posts.copy()
    engagement = posts[["likes", "reposts", "replies"]].fillna(0).sum(axis=1)
    followers = posts["followers_count"].fillna(0)

    weight = np.log1p(followers) * np.log1p(engagement)
    cap = weight.quantile(MAX_WEIGHT_PERCENTILE)
    posts["engagement_weight"] = weight.clip(upper=cap) if cap > 0 else weight
    return posts


def weighted_sentiment(posts: pd.DataFrame) -> float:
    """Engagement-weighted average sentiment for a group of posts."""
    if posts.empty or posts["engagement_weight"].sum() == 0:
        return posts["sentiment_score"].mean() if not posts.empty else 0.0
    return float((posts["sentiment_score"] * posts["engagement_weight"]).sum() / posts["engagement_weight"].sum())
