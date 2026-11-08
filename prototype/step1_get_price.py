import yfinance as yf
import pandas as pd

TICKER = "BTC-USD"

df = yf.download(TICKER, period="30d", interval="1h", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()
df = df.rename(columns={"Datetime": "time", "Close": "price"})
df = df[["time", "price"]]
df["price"] = df["price"].astype(float)

df.to_csv("price_data.csv", index=False)
print(f"Saved {len(df)} hourly price rows to price_data.csv")
print(df.head())
print(df.tail())
