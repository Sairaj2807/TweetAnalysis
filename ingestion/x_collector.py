"""Collects X posts about an asset using Apify, across ticker/name/contextual queries.

NOTE: Apify X/Twitter scraping actors vary in their exact input schema. The
`run_input` built below targets the common "searchTerms" / "maxItems" shape used
by most X search actors on the Apify Store. If the actor you pick (set via
APIFY_ACTOR_ID) expects different field names, adjust `_build_run_input` only --
everything downstream (normalization, dedup, DB storage) stays the same.
"""
from datetime import datetime, timezone

import pandas as pd

from config.settings import ASSET_CASHTAG, ASSET_NAME, ASSET_TICKER, HISTORY_DAYS
from ingestion.apify_client import run_actor_and_get_items

CONTEXTUAL_TERMS = [
    "bullish",
    "bearish",
    "price target",
    "to the moon",
    "crash",
    "breakout",
    "sell off",
]


def _build_queries() -> list[str]:
    base = ASSET_NAME.split()[0] if ASSET_NAME else ASSET_TICKER
    queries = [ASSET_CASHTAG, ASSET_NAME]
    queries += [f"{base} {term}" for term in CONTEXTUAL_TERMS]
    return queries


def _build_run_input(query: str, max_items: int = 500) -> dict:
    return {
        "searchTerms": [query],
        "maxItems": max_items,
        "sort": "Latest",
        "historyDays": HISTORY_DAYS,
    }


def collect_posts(asset: str = None, max_items_per_query: int = 500) -> pd.DataFrame:
    asset = asset or ASSET_TICKER
    all_rows = []
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for query in _build_queries():
        run_input = _build_run_input(query, max_items_per_query)
        items = run_actor_and_get_items(run_input)
        for item in items:
            all_rows.append(_normalize_item(item, query, asset, retrieved_at))

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["post_id"])
    return df


def _normalize_item(item: dict, query: str, asset: str, retrieved_at: str) -> dict:
    """Maps a raw Apify dataset item to our x_posts schema.

    Field names below are the common ones used by X search actors; adjust the
    .get(...) keys if your chosen actor's output differs (inspect one raw item
    first, e.g. with `python -c "import json,ingestion.x_collector as c; ..."`.)
    """
    return {
        "post_id": str(item.get("id") or item.get("tweetId") or item.get("url")),
        "created_at_utc": _to_utc_iso(item.get("createdAt") or item.get("timestamp")),
        "text": item.get("text") or item.get("fullText") or "",
        "username": item.get("author", {}).get("userName") if isinstance(item.get("author"), dict) else item.get("username"),
        "followers_count": _safe_int(
            item.get("author", {}).get("followers") if isinstance(item.get("author"), dict) else item.get("followersCount")
        ),
        "likes": _safe_int(item.get("likeCount") or item.get("likes")),
        "reposts": _safe_int(item.get("retweetCount") or item.get("reposts")),
        "views": _safe_int(item.get("viewCount") or item.get("views")),
        "replies": _safe_int(item.get("replyCount") or item.get("replies")),
        "search_query": query,
        "asset": asset,
        "retrieved_at_utc": retrieved_at,
    }


def _to_utc_iso(ts) -> str:
    if ts is None:
        return ""
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return pd.to_datetime(ts, utc=True).isoformat()


def _safe_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    df = collect_posts()
    print(f"Collected {len(df)} posts")
    out_path = "data/raw/x_posts.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
