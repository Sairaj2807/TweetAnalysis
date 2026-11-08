"""Builds the historical (2021-2023) price+sentiment training dataset from the
Kaggle tweet sample, since Apify can't backfill months of X history (see
ingestion/x_collector.py). Separate from main.py's live pipeline because this
uses daily candles (yfinance 1h data only goes back ~730 days) and a different
time period entirely -- see README "Known Limitations".
"""
import os

import pandas as pd
import yfinance as yf

from database.connection import init_db
from database.repository import upsert_dataframe
from features.feature_pipeline import HISTORICAL_FEATURE_SETS, add_targets, build_features, finalize_dataset
from ingestion.historical_kaggle_loader import load_sampled_historical_posts
from processing.aggregation import aggregate_posts_by_interval
from reporting.research_report import generate_evaluation_report
from sentiment.engagement_weighting import add_engagement_weight
from sentiment.sentiment_scoring import score_posts
from validation.metrics import PERIODS_PER_YEAR_HOURLY
from validation.model_comparison import run_comparison, summarize_by_feature_set_and_horizon
from visualization.sentiment_price import plot_sentiment_vs_price

ASSET = "BTC-USD"
SCORED_CACHE = "data/raw/historical_x_posts_scored.csv"


def build_window_dataset(posts: pd.DataFrame, asset_key: str) -> pd.DataFrame:
    """Turns one scored+weighted posts window into a finalized feature+target dataset.
    Kept per-window (not run on a concatenated multi-period posts frame) so that
    rolling features and shift(-h) targets never bridge a date gap between windows --
    see run_historical_backtest_combined.py, which calls this once per window and
    concatenates the already-finalized results."""
    start = posts["created_at_utc"].min().date()
    end = posts["created_at_utc"].max().date()
    print(f"  Fetching daily {ASSET} price bars for {start} -> {end}...")
    price = yf.download(ASSET, start=str(start), end=str(end + pd.Timedelta(days=1)), interval="1d", progress=False)
    if isinstance(price.columns, pd.MultiIndex):
        price.columns = price.columns.get_level_values(0)
    price = price.reset_index().rename(columns={"Date": "timestamp_utc", "Close": "price"})
    price["timestamp_utc"] = pd.to_datetime(price["timestamp_utc"], utc=True)
    price = price[["timestamp_utc", "price"]]
    print(f"    {len(price)} daily price bars")

    sentiment_agg = aggregate_posts_by_interval(posts, interval="1D")
    merged = price.merge(sentiment_agg, on="timestamp_utc", how="left")

    df = build_features(merged)
    df = add_targets(df)
    df = finalize_dataset(df)
    df["asset"] = asset_key
    print(f"    {len(df)} finalized rows for window {start} -> {end}")
    return df


def main():
    init_db()

    if os.path.exists(SCORED_CACHE):
        print(f"Loading cached scored posts from {SCORED_CACHE}...")
        posts = pd.read_csv(SCORED_CACHE)
        posts["created_at_utc"] = pd.to_datetime(posts["created_at_utc"], utc=True)
    else:
        print("Loading + sampling historical Kaggle posts...")
        posts = load_sampled_historical_posts(per_day_cap=200)
        print(f"  {len(posts)} posts across {posts['created_at_utc'].dt.date.nunique()} days")

        print("Scoring sentiment with FinBERT (this will take a few minutes)...")
        posts = score_posts(posts)
        posts.to_csv(SCORED_CACHE, index=False)
        print(f"  saved {SCORED_CACHE}")

    # engagement_weight always recomputed on top of the cache (cheap, no FinBERT
    # involved) so a fix to the weighting formula applies even to cached data.
    posts = add_engagement_weight(posts)

    print("Building features + 6-horizon targets (daily horizons: n+0..n+5 days)...")
    df = build_window_dataset(posts, ASSET + "_HIST")

    print(f"Final historical dataset: {len(df)} rows")
    upsert_dataframe(df, "feature_dataset", ["asset", "timestamp_utc"])
    df.to_csv("data/processed/historical_feature_dataset.csv", index=False)
    print("Saved to database (asset=BTC-USD_HIST) and data/processed/historical_feature_dataset.csv")

    plot_sentiment_vs_price(df, "data/reports/historical_sentiment_vs_price.png", title="BTC-USD Historical (2021-2023) Sentiment vs Price")

    print("Running walk-forward validation (n+1..n+5 DAYS ahead, daily candles)...")
    DAILY_PERIODS_PER_YEAR = 365
    results = run_comparison(df, n_splits=4, periods_per_year=DAILY_PERIODS_PER_YEAR, feature_sets=HISTORICAL_FEATURE_SETS)
    summary = summarize_by_feature_set_and_horizon(results)
    summary.to_csv("data/reports/historical_model_comparison_summary.csv", index=False)

    report = generate_evaluation_report(summary, "BTC-USD (2021-2023 historical backtest, daily candles)")
    with open("data/reports/historical_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved data/reports/historical_evaluation_report.md -- this is the real evaluation deliverable")


if __name__ == "__main__":
    main()
