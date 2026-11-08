"""Loads the second historical dataset (data/raw/tweets.csv). This file turned
out to span 2011-2019 (confirmed by scanning it), not just late 2019 as a
head/tail check initially suggested -- it's not sorted chronologically, and
roughly half its rows predate 2019. We restrict to [MIN_DATE, MAX_DATE) below:
MIN_DATE=2014-09-17 because that's yfinance's earliest available BTC-USD daily
bar (confirmed by testing) -- any tweet before that has no price to pair with,
so scoring it would be pure wasted FinBERT compute. MAX_DATE=2021-02-05 because
that's where the Kaggle dataset (historical_kaggle_loader.py) starts, so the
two windows stay non-overlapping when combined in run_historical_backtest_combined.py.

Schema: semicolon-delimited, columns id;user;fullname;url;timestamp;replies;
likes;retweets;text. No follower counts (unlike the Kaggle dataset, which has
followers but no per-tweet engagement) -- so this file fills followers_count
with 0 and keeps the real likes/retweets/replies counts. Retweets are
excluded via a simple "starts with RT @" heuristic since there's no explicit
is_retweet flag.
"""
import csv
import random
from datetime import datetime, timezone

import pandas as pd

csv.field_size_limit(2**31 - 1)

FILE = "data/raw/tweets.csv"
MIN_DATE = "2014-09-17"
MAX_DATE = "2021-02-05"


def load_sampled_2019_posts(per_day_cap: int = 200, seed: int = 42) -> pd.DataFrame:
    """Reservoir-samples up to `per_day_cap` real tweets per calendar day within
    [MIN_DATE, MAX_DATE), excluding retweets."""
    rng = random.Random(seed)
    reservoirs: dict[str, list] = {}
    seen_counts: dict[str, int] = {}

    with open(FILE, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        col_idx = {name: i for i, name in enumerate(c.strip().lstrip("﻿") for c in header)}
        needed = ["id", "user", "timestamp", "replies", "likes", "retweets", "text"]
        idx = {col: col_idx[col] for col in needed}

        for row in reader:
            if len(row) <= max(idx.values()):
                continue
            text = row[idx["text"]]
            if text.strip().upper().startswith("RT @") or text.strip().upper().startswith("RT@"):
                continue
            date_str = row[idx["timestamp"]]
            if len(date_str) < 10 or date_str[4] != "-" or date_str[7] != "-":
                continue
            if not (MIN_DATE <= date_str[:10] < MAX_DATE):
                continue

            day_key = date_str[:10]
            seen_counts[day_key] = seen_counts.get(day_key, 0) + 1
            n = seen_counts[day_key]

            record = (
                date_str,
                text,
                row[idx["user"]],
                row[idx["likes"]],
                row[idx["retweets"]],
                row[idx["replies"]],
            )

            bucket = reservoirs.setdefault(day_key, [])
            if len(bucket) < per_day_cap:
                bucket.append(record)
            else:
                j = rng.randint(1, n)
                if j <= per_day_cap:
                    bucket[j - 1] = record

    rows = []
    for day_key, bucket in reservoirs.items():
        for date_str, text, user, likes, retweets, replies in bucket:
            rows.append(
                {
                    "created_at_utc": date_str,
                    "text": text,
                    "username": user,
                    "likes": likes,
                    "reposts": retweets,
                    "replies": replies,
                }
            )

    df = pd.DataFrame(rows)
    df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_at_utc", "text"])
    df["text"] = df["text"].astype(str)
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    df["reposts"] = pd.to_numeric(df["reposts"], errors="coerce").fillna(0).astype(int)
    df["replies"] = pd.to_numeric(df["replies"], errors="coerce").fillna(0).astype(int)

    # no follower counts in this dataset -- unlike the Kaggle one, which has followers but no engagement
    df["followers_count"] = 0
    df["views"] = 0
    df["post_id"] = ["tweets2019_" + str(i) for i in range(len(df))]
    df["search_query"] = "historical_2019"
    df["asset"] = "BTC-USD"
    df["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()

    return df.sort_values("created_at_utc").reset_index(drop=True)


if __name__ == "__main__":
    df = load_sampled_2019_posts()
    print(f"Sampled {len(df)} posts across {df['created_at_utc'].dt.date.nunique()} days")
    print(f"Date range: {df['created_at_utc'].min()} -> {df['created_at_utc'].max()}")
    df.to_csv("data/raw/historical_tweets_csv_posts_sampled.csv", index=False)
