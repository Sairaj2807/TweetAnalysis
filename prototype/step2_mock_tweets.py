import pandas as pd
import numpy as np

# NOTE: This is MOCK tweet data standing in for a real X/Twitter scraper.
# Real scraping (snscrape/twscrape/Apify) is unreliable without auth/paid access.
# This proves the pipeline works end-to-end; swap this file's output for real
# scraped tweets later without changing any downstream step.

price = pd.read_csv("price_data.csv", parse_dates=["time"])

bullish = [
    "Bitcoin looking strong today, breaking resistance",
    "Huge buy volume on BTC, bullish signal",
    "BTC to the moon, accumulating more",
    "Great fundamentals for Bitcoin this week",
    "Institutional money flowing into BTC, very positive",
]
bearish = [
    "Bitcoin dumping hard, sell pressure increasing",
    "BTC breaking down, bearish structure forming",
    "Scared about this Bitcoin crash, selling my bags",
    "Bad news for crypto markets today, BTC down",
    "BTC looking weak, expecting further downside",
]
neutral = [
    "Bitcoin trading sideways today",
    "Watching BTC price action closely",
    "No major moves in Bitcoin right now",
    "BTC consolidating in a tight range",
]

rng = np.random.default_rng(42)
rows = []
for t in price["time"]:
    # a few posts per hour, random mix skewed slightly by random walk
    n_posts = rng.integers(1, 6)
    for _ in range(n_posts):
        bucket = rng.choice(["bull", "bear", "neutral"], p=[0.4, 0.3, 0.3])
        text = rng.choice(bullish if bucket == "bull" else bearish if bucket == "bear" else neutral)
        rows.append({"time": t, "text": text})

tweets = pd.DataFrame(rows)
tweets.to_csv("tweets_data.csv", index=False)
print(f"Saved {len(tweets)} mock tweet rows to tweets_data.csv")
print(tweets.head())
