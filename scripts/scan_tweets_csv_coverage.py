"""One-off analysis script answering Gilad's Q7: exact per-year tweet counts in
data/raw/tweets.csv, and how many days between 2014-09-17 and end of 2019 have no
tweets at all. Not part of any pipeline -- run manually when this information is
needed for a report/reply.

Run with: python scripts/scan_tweets_csv_coverage.py
"""
import csv
from collections import Counter
from datetime import date, timedelta

csv.field_size_limit(2**31 - 1)

FILE = "data/raw/tweets.csv"
RANGE_START = date(2014, 9, 17)
RANGE_END = date(2019, 12, 31)


def main():
    year_counts = Counter()
    days_with_tweets = set()
    total = 0

    with open(FILE, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        col_idx = {name: i for i, name in enumerate(c.strip().lstrip("﻿") for c in header)}
        ts_i = col_idx["timestamp"]

        for row in reader:
            total += 1
            if len(row) <= ts_i:
                continue
            d = row[ts_i]
            if len(d) < 10 or d[4] != "-" or d[7] != "-":
                continue
            year_counts[d[:4]] += 1
            day = d[:10]
            if "2014" <= day[:4] <= "2019":
                days_with_tweets.add(day)

    print(f"Total rows scanned: {total}")
    print("\nPer-year tweet counts (before sampling):")
    for y in sorted(year_counts):
        print(f"  {y}: {year_counts[y]:,}")

    n_days_in_range = (RANGE_END - RANGE_START).days + 1
    n_days_with_tweets = sum(1 for d in days_with_tweets if RANGE_START <= date.fromisoformat(d) <= RANGE_END)
    n_days_without = n_days_in_range - n_days_with_tweets
    print(f"\nDays from {RANGE_START} to {RANGE_END}: {n_days_in_range}")
    print(f"Days with at least one tweet: {n_days_with_tweets}")
    print(f"Days with ZERO tweets: {n_days_without}")


if __name__ == "__main__":
    main()
