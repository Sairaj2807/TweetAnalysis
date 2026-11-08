"""End-to-end demonstration of the full project objective:

  "Create a Python-based ML pipeline that predicts an asset's price at
  n+0, n+1, n+2, n+3, n+4 and n+5 based on X sentiment and price only.
  The system scrapes X posts, ingests recent news, joins the datasets in a
  database, and trains a predictive model using rigorous out-of-sample
  validation."

This script runs every stage in order and prints what's happening at each
step, so each requirement can be checked off live:

  1. Scrape X posts (Apify)                         -> ingestion/x_collector.py
  2. Ingest recent news (Tavily)                     -> ingestion/tavily_client.py
  3. Join datasets in a database                     -> database/ (SQLite)
  4. Feature engineering (price + sentiment only)    -> features/
  5. Train predictive model, walk-forward validation -> validation/
  6. Produce the n+0..n+5 price prediction           -> models/inference.py  <-- the actual deliverable

Run with: python demo_end_to_end.py
"""
import pandas as pd

from config.settings import ASSET_TICKER, HORIZONS
from database.connection import init_db
from database.repository import read_table
from features.feature_pipeline import FEATURE_SETS
from models.inference import generate_prediction_matrix
from models.multi_horizon import train_multi_horizon
from models.xgboost_model import build_xgboost_model
from processing.dataset_builder import build_and_store_dataset
from validation.leakage_checks import assert_no_overlap
from validation.metrics import forecast_metrics
from validation.walk_forward import chronological_split
from visualization.prediction_chart import plot_actual_vs_predicted
from visualization.prediction_matrix import render_prediction_matrix_table


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    section("STEP 1-2: Scrape X posts + ingest news, store real data")
    import main as live_pipeline  # reuses main.py's ingest_price/ingest_posts/ingest_news

    init_db()
    live_pipeline.ingest_price()
    posts = live_pipeline.ingest_posts(use_cache=True)  # uses cached real Apify data if present
    news = live_pipeline.ingest_news()
    print(f"  -> {len(posts)} real X posts in database, {len(news)} news articles")

    section("STEP 3: Confirm the join -- posts, news and price are in one database")
    x_posts_db = read_table("x_posts")
    price_db = read_table("price_bars")
    news_db = read_table("news_articles")
    print(f"  x_posts table:    {len(x_posts_db)} rows")
    print(f"  price_bars table: {len(price_db)} rows")
    print(f"  news_articles table: {len(news_db)} rows")
    print("  (all three live in data/sentiment_price.db, joined on timestamp by dataset_builder.py)")

    section("STEP 4: Feature engineering (price + sentiment only) + join into one dataset")
    dataset = build_and_store_dataset()
    feature_dataset_db = read_table("feature_dataset", "asset = ?", (ASSET_TICKER,))
    print(f"  Built {len(dataset)} feature rows, confirmed {len(feature_dataset_db)} rows in feature_dataset table")
    print(f"  Feature sets available: {list(FEATURE_SETS.keys())}")
    print(f"  Sample row:\n{dataset.iloc[-1][['timestamp_utc', 'price', 'sentiment_mean', 'news_sentiment']].to_string()}")

    section("STEP 5: Train model + rigorous out-of-sample (walk-forward) validation")
    feature_cols = FEATURE_SETS["price_plus_sentiment"]
    train, val, test = chronological_split(dataset)
    assert_no_overlap(train, test)  # proves no data leakage between train and test
    print(f"  Train: {len(train)} rows | Val: {len(val)} rows | Test (out-of-sample): {len(test)} rows")
    print(f"  Train window ends: {train['timestamp_utc'].max()}")
    print(f"  Test window starts: {test['timestamp_utc'].min()}  (strictly after training data -- no leakage)")

    models = train_multi_horizon(train, feature_cols, build_xgboost_model)
    print(f"  Trained {len(models)} models (one per horizon: {list(models.keys())})")

    from models.multi_horizon import predict_multi_horizon
    test_preds = predict_multi_horizon(models, test, feature_cols)
    print("\n  Out-of-sample performance (on data the model never saw during training):")
    for h in HORIZONS:
        if h == 0:
            continue
        m = forecast_metrics(test[f"target_n{h}"].values, test_preds[f"n{h}"].values, test["price"].values)
        print(f"    n+{h}: MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  Direction Accuracy={m['direction_accuracy']:.1%}")

    plot_actual_vs_predicted(
        test, test_preds, "data/reports/live_actual_vs_predicted.png",
        title=f"{ASSET_TICKER} Actual vs Predicted (out-of-sample)",
    )
    print("  Saved data/reports/live_actual_vs_predicted.png")

    section("STEP 6: THE DELIVERABLE -- predict price at n+0, n+1, n+2, n+3, n+4, n+5")
    final_models = train_multi_horizon(dataset, feature_cols, build_xgboost_model)
    latest_row = dataset.tail(1)
    matrix = generate_prediction_matrix(final_models, latest_row, feature_cols)
    print(f"\n  {ASSET_TICKER} price forecast as of {latest_row['timestamp_utc'].iloc[0]}:\n")
    print(matrix.to_string(index=False))

    matrix.to_csv("data/reports/live_prediction_matrix.csv", index=False)
    print("\n  Saved to data/reports/live_prediction_matrix.csv")

    render_prediction_matrix_table(matrix, "data/reports/live_prediction_matrix.png", ASSET_TICKER)
    print("  Saved data/reports/live_prediction_matrix.png (visual matrix output)")

    section("DONE -- all 6 pipeline stages demonstrated end to end")


if __name__ == "__main__":
    main()
