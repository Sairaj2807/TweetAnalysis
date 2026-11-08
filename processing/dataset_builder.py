import pandas as pd

from config.settings import ASSET_TICKER
from database.repository import read_table, upsert_dataframe
from features.feature_pipeline import add_targets, build_features, finalize_dataset
from processing.aggregation import aggregate_news_by_interval, aggregate_posts_by_interval


def build_and_store_dataset(asset: str = None) -> pd.DataFrame:
    asset = asset or ASSET_TICKER

    posts = read_table("x_posts", "asset = ?", (asset,))
    news = read_table("news_articles", "asset = ?", (asset,))
    price = read_table("price_bars", "asset = ?", (asset,))

    if price.empty:
        raise RuntimeError(f"No price bars found for {asset}; run ingestion/price_collector.py first")

    price = price.rename(columns={"close": "price"})[["timestamp_utc", "price"]]
    price["timestamp_utc"] = pd.to_datetime(price["timestamp_utc"], utc=True)

    sentiment_agg = aggregate_posts_by_interval(posts) if not posts.empty else pd.DataFrame()
    news_agg = aggregate_news_by_interval(news) if not news.empty else pd.DataFrame()

    merged = price.merge(sentiment_agg, on="timestamp_utc", how="left") if not sentiment_agg.empty else price
    if not news_agg.empty:
        merged = merged.merge(news_agg, on="timestamp_utc", how="left")

    df = build_features(merged)
    df = add_targets(df)
    df = finalize_dataset(df)
    df["asset"] = asset

    upsert_dataframe(df, "feature_dataset", ["asset", "timestamp_utc"])
    return df


if __name__ == "__main__":
    df = build_and_store_dataset()
    print(f"Built feature_dataset with {len(df)} rows")
