"""End-to-end pipeline: ingest -> sentiment score -> store -> build features -> validate -> report.

Run with `python main.py`. Apify/Tavily steps are skipped gracefully (with a
warning) if their API keys are not yet set in .env -- price-only data still
flows through the whole pipeline so it's runnable before credentials exist.
"""
import os

import pandas as pd

from config.settings import ASSET_TICKER
from database.connection import init_db
from database.repository import upsert_dataframe
from processing.cleaning import clean_posts
from processing.deduplication import dedupe_posts
from processing.dataset_builder import build_and_store_dataset
from reporting.research_report import generate_evaluation_report
from reporting.sentiment_report import generate_sentiment_report
from sentiment.sentiment_scoring import score_posts
from sentiment.engagement_weighting import add_engagement_weight
from sentiment.news_sentiment import score_news
from validation.model_comparison import run_comparison, summarize_by_feature_set_and_horizon
from visualization.heatmap import plot_sentiment_heatmap
from visualization.sentiment_price import plot_sentiment_vs_price

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)


def ingest_price():
    from ingestion.price_collector import collect_price_bars

    df = collect_price_bars()
    upsert_dataframe(df, "price_bars", ["asset", "timestamp_utc"])
    print(f"[price] stored {len(df)} bars")


def ingest_posts():
    try:
        from ingestion.x_collector import collect_posts

        posts = collect_posts()
    except RuntimeError as e:
        print(f"[x_posts] SKIPPED ({e}) -- add APIFY_API_TOKEN/APIFY_ACTOR_ID to .env to enable")
        return pd.DataFrame()

    if posts.empty:
        print("[x_posts] no posts returned")
        return posts

    posts = clean_posts(dedupe_posts(posts))
    posts = score_posts(posts)
    posts = add_engagement_weight(posts)
    upsert_dataframe(posts, "x_posts", ["post_id"])
    print(f"[x_posts] stored {len(posts)} posts")
    return posts


def ingest_news():
    try:
        from ingestion.tavily_client import collect_recent_news

        news = collect_recent_news()
    except RuntimeError as e:
        print(f"[news] SKIPPED ({e}) -- add TAVILY_API_KEY to .env to enable")
        return pd.DataFrame()

    if news.empty:
        print("[news] no articles returned")
        return news

    news = score_news(news)
    upsert_dataframe(news, "news_articles", ["url"])
    print(f"[news] stored {len(news)} articles")
    return news


def main():
    init_db()
    ingest_price()
    posts = ingest_posts()
    news = ingest_news()

    dataset = build_and_store_dataset()
    print(f"[dataset] built {len(dataset)} feature rows")

    if len(dataset) < 50:
        print("[WARNING] Very little data to validate on -- results below are not meaningful yet.")

    plot_sentiment_vs_price(dataset, "data/reports/sentiment_vs_price.png", title=f"{ASSET_TICKER} Sentiment vs Price")
    if not posts.empty:
        plot_sentiment_heatmap(dataset, "data/reports/sentiment_heatmap.png", title=f"{ASSET_TICKER} Sentiment Heatmap")

    sentiment_report = generate_sentiment_report(posts, news, ASSET_TICKER)
    with open("data/reports/sentiment_report.md", "w", encoding="utf-8") as f:
        f.write(sentiment_report)
    print("[report] sentiment_report.md written")

    results = run_comparison(dataset, n_splits=4)
    if results.empty:
        print("[WARNING] Not enough data for walk-forward validation folds yet.")
        return

    summary = summarize_by_feature_set_and_horizon(results)
    summary.to_csv("data/reports/model_comparison_summary.csv", index=False)

    eval_report = generate_evaluation_report(summary, ASSET_TICKER)
    with open("data/reports/evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(eval_report)
    print("[report] evaluation_report.md written")


if __name__ == "__main__":
    main()
