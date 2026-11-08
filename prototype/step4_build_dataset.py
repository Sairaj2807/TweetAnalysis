import pandas as pd

price = pd.read_csv("price_data.csv", parse_dates=["time"])
tweets = pd.read_csv("tweets_scored.csv", parse_dates=["time"])

# Step 4: average sentiment per hour
sentiment_hourly = tweets.groupby("time")["sentiment"].mean().reset_index()
sentiment_hourly = sentiment_hourly.rename(columns={"sentiment": "sentiment_mean"})

data = price.merge(sentiment_hourly, on="time", how="left")
data["sentiment_mean"] = data["sentiment_mean"].fillna(0)

# Step 5: target = price 1 hour later
data = data.sort_values("time").reset_index(drop=True)
data["target_price_next"] = data["price"].shift(-1)

# drop last row (no future price to predict)
data = data.dropna(subset=["target_price_next"]).reset_index(drop=True)

data.to_csv("final_dataset.csv", index=False)
print(f"Saved {len(data)} rows to final_dataset.csv")
print(data.head())
