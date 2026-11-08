"""Thin wrapper around the Apify Python client for running X/Twitter scraping actors.

Requires APIFY_API_TOKEN and APIFY_ACTOR_ID in .env. See README for which actor
we used (e.g. an X/Twitter search scraper actor from Apify Store) and why.
"""
from apify_client import ApifyClient

from config.settings import APIFY_API_TOKEN, APIFY_ACTOR_ID


def get_client() -> ApifyClient:
    if not APIFY_API_TOKEN:
        raise RuntimeError("APIFY_API_TOKEN is not set in .env")
    return ApifyClient(APIFY_API_TOKEN)


def run_actor_and_get_items(run_input: dict, actor_id: str = None) -> list[dict]:
    """Runs the configured Apify actor synchronously and returns its dataset items."""
    client = get_client()
    actor_id = actor_id or APIFY_ACTOR_ID
    if not actor_id:
        raise RuntimeError("APIFY_ACTOR_ID is not set in .env")

    run = client.actor(actor_id).call(run_input=run_input)
    dataset_id = run["defaultDatasetId"]

    items = []
    for item in client.dataset(dataset_id).iterate_items():
        items.append(item)
    return items
