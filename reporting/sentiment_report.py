from datetime import datetime, timezone

import pandas as pd


def generate_sentiment_report(posts: pd.DataFrame, news: pd.DataFrame, asset: str) -> str:
    """Builds the one-page current sentiment outlook as Markdown."""
    now = datetime.now(timezone.utc).isoformat()

    if posts.empty:
        return f"# {asset} Sentiment Report\n\nNo posts available.\n"

    overall = posts["sentiment_score"].mean()
    pos_pct = (posts["sentiment_score"] > 0.1).mean() * 100
    neg_pct = (posts["sentiment_score"] < -0.1).mean() * 100
    neu_pct = 100 - pos_pct - neg_pct
    total_views = posts["views"].fillna(0).sum() if "views" in posts.columns else 0

    news_sentiment = news["sentiment_score"].mean() if not news.empty else None

    lines = [
        f"# {asset} — Current Sentiment Outlook",
        "",
        f"Report generated: {now}",
        "",
        f"## Overall sentiment: {overall:+.2f}",
        "",
        f"- Positive: {pos_pct:.1f}%",
        f"- Negative: {neg_pct:.1f}%",
        f"- Neutral: {neu_pct:.1f}%",
        "",
        "## Social activity",
        f"- Posts analyzed: {len(posts):,}",
        f"- Total views (if available): {total_views:,.0f}",
        "",
        "## News sentiment",
        f"- Average news sentiment: {news_sentiment:+.2f}" if news_sentiment is not None else "- No recent news retrieved",
        "",
        "## Interpretation",
        (
            "The current social sentiment environment is "
            + ("positive" if overall > 0.1 else "negative" if overall < -0.1 else "mixed/neutral")
            + ". Sentiment alone does not establish a price forecast -- see the "
            "model evaluation report for out-of-sample evidence of predictive value."
        ),
    ]
    return "\n".join(lines)
