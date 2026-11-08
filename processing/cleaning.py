import pandas as pd


def clean_posts(posts: pd.DataFrame) -> pd.DataFrame:
    posts = posts.copy()
    posts = posts[posts["text"].str.strip().astype(bool)]
    posts = posts.dropna(subset=["created_at_utc"])
    for col in ["likes", "reposts", "views", "replies", "followers_count"]:
        if col in posts.columns:
            posts[col] = posts[col].fillna(0)
    return posts
