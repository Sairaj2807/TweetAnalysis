import numpy as np
import pandas as pd

MAX_WEIGHT_PERCENTILE = 0.99  # cap extreme weights so a few large accounts don't dominate


def add_engagement_weight(posts: pd.DataFrame) -> pd.DataFrame:
    """Weights each post by engagement, preferring whichever signal this source
    actually has. Some historical tweet sources only have follower counts (no
    per-tweet likes/retweets/replies), others only have per-tweet engagement (no
    follower counts) -- multiplying log1p(followers) * log1p(engagement) as a
    single formula silently collapses to 0 for every row whenever either side of
    the product is always 0, which makes sentiment_weighted identical to
    sentiment_mean without any error or warning. Falling back to whichever signal
    is actually present keeps weighting meaningful for every source.
    """
    posts = posts.copy()
    engagement = posts[["likes", "reposts", "replies"]].fillna(0).sum(axis=1)
    followers = posts["followers_count"].fillna(0)

    has_engagement = engagement.sum() > 0
    has_followers = followers.sum() > 0
    if has_engagement and has_followers:
        weight = np.log1p(followers) * np.log1p(engagement)
    elif has_engagement:
        weight = np.log1p(engagement)
    elif has_followers:
        weight = np.log1p(followers)
    else:
        weight = pd.Series(0.0, index=posts.index)

    cap = weight.quantile(MAX_WEIGHT_PERCENTILE)
    posts["engagement_weight"] = weight.clip(upper=cap) if cap > 0 else weight
    return posts


def weighted_sentiment(posts: pd.DataFrame) -> float:
    """Engagement-weighted average sentiment for a group of posts."""
    if posts.empty or posts["engagement_weight"].sum() == 0:
        return posts["sentiment_score"].mean() if not posts.empty else 0.0
    return float((posts["sentiment_score"] * posts["engagement_weight"]).sum() / posts["engagement_weight"].sum())
