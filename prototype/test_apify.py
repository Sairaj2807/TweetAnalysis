"""One-off cheap smoke test for the apidojo/tweet-scraper actor before scaling up.
Run from repo root: python prototype/test_apify.py
"""
from ingestion.apify_client import run_actor_and_get_items
from ingestion.x_collector import _build_run_input

run_input = _build_run_input("$BTC", max_items=20)
print("Run input:", run_input)

items = run_actor_and_get_items(run_input)
print(f"\nGot {len(items)} items")

if items:
    import json
    print("\nFirst raw item (to verify field names):")
    print(json.dumps(items[0], indent=2, default=str)[:2000])
