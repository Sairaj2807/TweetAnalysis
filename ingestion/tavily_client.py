"""Fetches recent news about the asset using Tavily's search API.

Requires TAVILY_API_KEY in .env. Tavily's search only returns currently-indexed
(recent) results -- it cannot retrieve news as it existed at an arbitrary past
timestamp, so this is used for the current sentiment report, not retroactively
inserted into historical training data (see README "Known Limitations").
"""
from datetime import datetime, timezone

import pandas as pd
from tavily import TavilyClient

from config.settings import ASSET_NAME, ASSET_TICKER, TAVILY_API_KEY


def get_client() -> TavilyClient:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set in .env")
    return TavilyClient(api_key=TAVILY_API_KEY)


def collect_recent_news(asset_name: str = None, max_results: int = 20) -> pd.DataFrame:
    asset_name = asset_name or ASSET_NAME
    client = get_client()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    response = client.search(
        query=f"{asset_name} price news",
        topic="news",
        search_depth="advanced",
        max_results=max_results,
    )

    rows = []
    for item in response.get("results", []):
        rows.append(
            {
                "url": item.get("url", ""),
                "published_at_utc": item.get("published_date", ""),
                "headline": item.get("title", ""),
                "source": item.get("url", "").split("/")[2] if item.get("url") else "",
                "summary": item.get("content", ""),
                "asset": asset_name,
                "retrieved_at_utc": retrieved_at,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = collect_recent_news()
    print(f"Collected {len(df)} news articles")
    out_path = "data/raw/news.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
