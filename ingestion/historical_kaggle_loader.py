"""Loads the historical Bitcoin tweets Kaggle dataset (kaushiksuresh147/bitcoin-tweets)
as the historical half of the training data, since Apify cannot reliably backfill
months of X history (see ingestion/x_collector.py module docstring).

Dataset covers 2021-02-05 to 2023-01-09, ~5.6M rows across two CSV files,
columns: user_name, user_location, user_description, user_created,
user_followers, user_friends, user_favourites, user_verified, date, text,
hashtags, source, is_retweet. No per-tweet engagement counts (likes/retweets)
-- that's a real limitation of this dataset versus live X data, noted in
README. Scoring all ~5.6M tweets with FinBERT would take hours; we instead
take an unbiased stratified sample (reservoir sampling per calendar day) so
every day in the 2-year span gets roughly equal representation.
"""
import csv
import random
import sys
from datetime import datetime, timezone

import pandas as pd

csv.field_size_limit(2**31 - 1)

FILES = [
    "data/raw/Bitcoin_tweets.csv",
    "data/raw/Bitcoin_tweets_dataset_2.csv",
]


def load_sampled_historical_posts(per_day_cap: int = 150, seed: int = 42) -> pd.DataFrame:
    """Reservoir-samples up to `per_day_cap` real tweets per calendar day across
    both Kaggle files, excluding retweets (to avoid duplicate-counting sentiment)."""
    rng = random.Random(seed)
    reservoirs: dict[str, list] = {}
    seen_counts: dict[str, int] = {}

    for fname in FILES:
        with open(fname, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            idx = {col: header.index(col) for col in ["date", "text", "user_followers", "is_retweet"]}

            for row in reader:
                if len(row) <= max(idx.values()):
                    continue
                date_str = row[idx["date"]]
                if len(date_str) < 10 or date_str[4] != "-" or date_str[7] != "-":
                    continue
                if row[idx["is_retweet"]].strip().lower() == "true":
                    continue

                day_key = date_str[:10]
                seen_counts[day_key] = seen_counts.get(day_key, 0) + 1
                n = seen_counts[day_key]

                record = (date_str, row[idx["text"]], row[idx["user_followers"]])

                bucket = reservoirs.setdefault(day_key, [])
                if len(bucket) < per_day_cap:
                    bucket.append(record)
                else:
                    j = rng.randint(1, n)
                    if j <= per_day_cap:
                        bucket[j - 1] = record

    rows = []
    for day_key, bucket in reservoirs.items():
        for date_str, text, followers in bucket:
            rows.append({"created_at_utc": date_str, "text": text, "followers_count": followers})

    df = pd.DataFrame(rows)
    df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_at_utc", "text"])
    df["text"] = df["text"].astype(str)
    df["followers_count"] = pd.to_numeric(df["followers_count"], errors="coerce").fillna(0).astype(int)

    # columns not present in this dataset -- filled with 0 so it matches the x_posts schema
    for col in ["likes", "reposts", "views", "replies"]:
        df[col] = 0
    df["post_id"] = ["kaggle_" + str(i) for i in range(len(df))]
    df["username"] = ""
    df["search_query"] = "historical_kaggle"
    df["asset"] = "BTC-USD"
    df["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()

    return df.sort_values("created_at_utc").reset_index(drop=True)


if __name__ == "__main__":
    df = load_sampled_historical_posts()
    print(f"Sampled {len(df)} historical posts across {df['created_at_utc'].dt.date.nunique()} days")
    print(f"Date range: {df['created_at_utc'].min()} -> {df['created_at_utc'].max()}")
    df.to_csv("data/raw/historical_x_posts_sampled.csv", index=False)
