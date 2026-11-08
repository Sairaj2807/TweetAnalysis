import pandas as pd


def dedupe_posts(posts: pd.DataFrame) -> pd.DataFrame:
    """Dedupe posts collected across multiple search queries, keeping the first search_query seen."""
    return posts.drop_duplicates(subset=["post_id"], keep="first").reset_index(drop=True)
