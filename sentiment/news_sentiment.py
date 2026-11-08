import pandas as pd

from sentiment.finbert import score_texts


def score_news(news: pd.DataFrame) -> pd.DataFrame:
    """Scores news headline+summary text with FinBERT, same scale as X post sentiment."""
    news = news.copy()
    if news.empty:
        news["sentiment_score"] = []
        return news

    combined_text = (news["headline"].fillna("") + ". " + news["summary"].fillna("")).tolist()
    probs = score_texts(combined_text)
    news["sentiment_score"] = [p[0] - p[1] for p in probs]
    return news
