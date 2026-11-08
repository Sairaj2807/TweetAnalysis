import pandas as pd

from sentiment.finbert import score_texts


def score_posts(posts: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Adds sentiment_positive/negative/neutral/score columns to a posts dataframe."""
    posts = posts.copy()
    texts = posts[text_col].fillna("").tolist()
    probs = score_texts(texts)

    posts["sentiment_positive"] = [p[0] for p in probs]
    posts["sentiment_negative"] = [p[1] for p in probs]
    posts["sentiment_neutral"] = [p[2] for p in probs]
    posts["sentiment_score"] = posts["sentiment_positive"] - posts["sentiment_negative"]
    return posts
