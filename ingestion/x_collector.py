"""Collects X posts about an asset using Apify's kaitoeasyapi tweet-scraper actor.

Actor: kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest
Chosen over apidojo/tweet-scraper because the apidojo actor's Free-Plan account
tier hard-caps every run at 10 items regardless of maxItems or available credit;
this actor has no such cap (confirmed: 300 real items in one unfiltered call).
Pay-per-result, ~$0.18-0.25 per 1,000 tweets.

IMPORTANT, CONFIRMED LIMITATION (tested 2026-11-08): this actor's since_time/
until_time historical date filters do NOT reliably work for windows beyond the
very recent past. Requesting a window from months ago still returns today's
tweets, and pads any shortfall with fabricated placeholder items (id="-1",
createdAt=None, text that looks plausible but is synthetic) rather than
returning fewer real results. This is true of the apidojo actor too (its docs
separately warn date-filtered queries "may return incomplete or empty
results"). Conclusion: cheap/free-tier Apify X actors cannot reliably backfill
months of history -- this is a real, actor-side data-availability limit, not a
bug in our code. See README "Known Limitations" for how this project handles
historical data instead (a pre-existing historical tweets dataset).

Given that, this collector is used for CURRENT/recent real sentiment only
(the "one-page current sentiment outlook" requirement, and to seed the live
dataset going forward) -- not for deep historical backfill. Every item is
checked against `_is_mock_item` and dropped if it is a fabricated placeholder.

Field schema (confirmed from a live sample item, 2026-11-08): id, text,
fullText, createdAt (e.g. "Sun Nov 08 05:02:22 +0000 2026"), likeCount,
retweetCount, viewCount, replyCount, author.userName, author.followers.
"""
from datetime import datetime, timezone

import pandas as pd

from config.settings import ASSET_CASHTAG, ASSET_NAME, ASSET_TICKER
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


def _build_run_input(query: str, max_items: int) -> dict:
    return {
        "searchTerms": [query],
        "maxItems": max_items,
        "queryType": "Latest",
        "lang": "en",
    }


def _is_mock_item(item: dict) -> bool:
    """This actor pads thin result sets with fake placeholder tweets (id=-1,
    createdAt=None) rather than returning fewer real results. Must be filtered
    out or they'll pollute sentiment scoring with fabricated text."""
    return str(item.get("id")) == "-1" or not item.get("createdAt")


def collect_posts(asset: str = None, max_items_per_query: int = 200) -> pd.DataFrame:
    """Collects current/recent real posts across all queries. See module
    docstring: this does NOT reach back months -- it captures what's available
    right now, which is what this actor can actually deliver reliably."""
    asset = asset or ASSET_TICKER
    retrieved_at = datetime.now(timezone.utc).isoformat()

    all_rows = []
    mock_dropped = 0
    queries = _build_queries()

    for i, query in enumerate(queries, 1):
        run_input = _build_run_input(query, max_items_per_query)
        try:
            items = run_actor_and_get_items(run_input)
        except Exception as e:
            print(f"  [{i}/{len(queries)}] FAILED for '{query}': {e}")
            continue

        real_items = []
        for item in items:
            if _is_mock_item(item):
                mock_dropped += 1
            else:
                real_items.append(item)

        print(f"  [{i}/{len(queries)}] '{query}': {len(real_items)} real items ({len(items) - len(real_items)} mock dropped)")
        for item in real_items:
            all_rows.append(_normalize_item(item, query, asset, retrieved_at))

    if mock_dropped:
        print(f"  Total mock/fabricated items dropped across all queries: {mock_dropped}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["post_id"])
    return df


def _normalize_item(item: dict, query: str, asset: str, retrieved_at: str) -> dict:
    author = item.get("author") or {}
    return {
        "post_id": str(item.get("id") or item.get("url")),
        "created_at_utc": _to_utc_iso(item.get("createdAt")),
        "text": item.get("fullText") or item.get("text") or "",
        "username": author.get("userName"),
        "followers_count": _safe_int(author.get("followers")),
        "likes": _safe_int(item.get("likeCount")),
        "reposts": _safe_int(item.get("retweetCount")),
        "views": _safe_int(item.get("viewCount")),
        "replies": _safe_int(item.get("replyCount")),
        "search_query": query,
        "asset": asset,
        "retrieved_at_utc": retrieved_at,
    }


def _to_utc_iso(ts) -> str:
    if ts is None or ts == "":
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
    print(f"\nCollected {len(df)} total real posts")
    out_path = "data/raw/x_posts.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
