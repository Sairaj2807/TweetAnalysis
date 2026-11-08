"""Combined historical backtest across two non-overlapping periods:
2014-09-17..2021-02-05 (data/raw/tweets.csv, filtered -- see
ingestion/historical_tweets_2019_loader.py for why: that file turned out to
span 2011-2019, not just 2019, so it's clipped to start at yfinance's
earliest BTC-USD daily bar and end where the Kaggle window begins) and
2021-02-05..2023-01-09/03-05 (the Kaggle dataset, see
ingestion/historical_kaggle_loader.py). Each window is scored, aggregated,
feature-engineered and target-shifted *separately* via
run_historical_backtest.build_window_dataset(), then the two already-finalized
frames are concatenated -- this guarantees rolling features and n+1..n+5
shift targets never bridge the boundary between the two periods.

The combined dataset gives walk-forward folds more rows than either period
alone, and spans multiple BTC market regimes (2014-2020 early-market years,
the 2021 bull peak, the 2022 crash) -- see README "Known limitations" for why
this is a more convincing test of the "does sentiment add value" question
than a single, smaller, single-regime sample.
"""
import pandas as pd

from database.connection import init_db
from database.repository import upsert_dataframe
from ingestion.historical_kaggle_loader import load_sampled_historical_posts
from ingestion.historical_tweets_2019_loader import load_sampled_2019_posts
from reporting.research_report import generate_evaluation_report
from run_historical_backtest import build_window_dataset
from sentiment.engagement_weighting import add_engagement_weight
from sentiment.sentiment_scoring import score_posts
from validation.model_comparison import run_comparison, summarize_by_feature_set_and_horizon
from visualization.sentiment_price import plot_sentiment_vs_price

ASSET_KEY = "BTC-USD_HIST_COMBINED"
DAILY_PERIODS_PER_YEAR = 365


def main():
    init_db()

    print("Loading + sampling 2014-2021 posts (data/raw/tweets.csv, date-filtered)...")
    posts_early = load_sampled_2019_posts(per_day_cap=200)
    print(f"  {len(posts_early)} posts across {posts_early['created_at_utc'].dt.date.nunique()} days")
    print(f"  Date range: {posts_early['created_at_utc'].min()} -> {posts_early['created_at_utc'].max()}")

    print("Loading + sampling 2021-2023 Kaggle posts...")
    posts_kaggle = load_sampled_historical_posts(per_day_cap=200)
    print(f"  {len(posts_kaggle)} posts across {posts_kaggle['created_at_utc'].dt.date.nunique()} days")

    print("Scoring both windows' sentiment with FinBERT (this will take a while)...")
    posts_early = score_posts(posts_early)
    posts_early = add_engagement_weight(posts_early)
    posts_kaggle = score_posts(posts_kaggle)
    posts_kaggle = add_engagement_weight(posts_kaggle)

    print("Building each window's features + targets separately (no cross-boundary leakage)...")
    print("Window 1: 2014-2021")
    df_early = build_window_dataset(posts_early, ASSET_KEY)
    print("Window 2: 2021-2023")
    df_kaggle = build_window_dataset(posts_kaggle, ASSET_KEY)

    df = pd.concat([df_early, df_kaggle], ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
    print(f"Combined dataset: {len(df)} rows ({len(df_early)} from 2014-2021 + {len(df_kaggle)} from 2021-2023)")

    upsert_dataframe(df, "feature_dataset", ["asset", "timestamp_utc"])
    df.to_csv("data/processed/historical_feature_dataset_combined.csv", index=False)
    print(f"Saved to database (asset={ASSET_KEY}) and data/processed/historical_feature_dataset_combined.csv")

    plot_sentiment_vs_price(
        df, "data/reports/historical_sentiment_vs_price_combined.png",
        title="BTC-USD Combined Historical (2019 + 2021-2023) Sentiment vs Price",
    )

    print("Running walk-forward validation (n+1..n+5 DAYS ahead, daily candles)...")
    results = run_comparison(df, n_splits=4, periods_per_year=DAILY_PERIODS_PER_YEAR)
    summary = summarize_by_feature_set_and_horizon(results)
    summary.to_csv("data/reports/historical_model_comparison_summary_combined.csv", index=False)

    report = generate_evaluation_report(summary, "BTC-USD (combined 2019 + 2021-2023 historical backtest, daily candles)")
    with open("data/reports/historical_evaluation_report_combined.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved data/reports/historical_evaluation_report_combined.md")


if __name__ == "__main__":
    main()
