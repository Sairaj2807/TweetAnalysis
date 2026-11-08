"""Collects historical OHLCV price bars via yfinance (free, no key required).

yfinance's intraday history is capped by Yahoo Finance itself (1h bars: ~730 days
max, but reliable continuous coverage is better over 60-day chunks). For the full
180-day window we request in chunks and concatenate.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from config.settings import ASSET_TICKER, CANDLE_INTERVAL, HISTORY_DAYS


def collect_price_bars(ticker: str = None, days: int = None, interval: str = None) -> pd.DataFrame:
    ticker = ticker or ASSET_TICKER
    days = days or HISTORY_DAYS
    interval = interval or CANDLE_INTERVAL

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    chunk_days = 59 if interval.endswith("h") or interval.endswith("m") else days
    frames = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        df = yf.download(
            ticker,
            start=chunk_start,
            end=chunk_end,
            interval=interval,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            frames.append(df)
        chunk_start = chunk_end

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames)
    full = full[~full.index.duplicated(keep="first")]
    full = full.sort_index().reset_index()
    full = full.rename(columns={full.columns[0]: "timestamp_utc"})
    full["timestamp_utc"] = pd.to_datetime(full["timestamp_utc"], utc=True)
    full = full.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    full["asset"] = ticker
    return full[["asset", "timestamp_utc", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    df = collect_price_bars()
    print(f"Collected {len(df)} price bars")
    out_path = "data/raw/price_bars.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
