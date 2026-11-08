CREATE TABLE IF NOT EXISTS x_posts (
    post_id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    text TEXT NOT NULL,
    username TEXT,
    followers_count INTEGER,
    likes INTEGER,
    reposts INTEGER,
    views INTEGER,
    replies INTEGER,
    search_query TEXT,
    asset TEXT,
    retrieved_at_utc TEXT,
    sentiment_positive REAL,
    sentiment_negative REAL,
    sentiment_neutral REAL,
    sentiment_score REAL,
    engagement_weight REAL
);

CREATE TABLE IF NOT EXISTS news_articles (
    url TEXT PRIMARY KEY,
    published_at_utc TEXT,
    headline TEXT,
    source TEXT,
    summary TEXT,
    asset TEXT,
    sentiment_score REAL,
    retrieved_at_utc TEXT
);

CREATE TABLE IF NOT EXISTS price_bars (
    asset TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    PRIMARY KEY (asset, timestamp_utc)
);

CREATE TABLE IF NOT EXISTS feature_dataset (
    asset TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    price REAL,
    sentiment_mean REAL,
    sentiment_weighted REAL,
    positive_ratio REAL,
    negative_ratio REAL,
    post_count INTEGER,
    engagement_total REAL,
    sentiment_change REAL,
    sentiment_lag_1 REAL,
    news_sentiment REAL,
    return_1 REAL,
    return_3 REAL,
    return_5 REAL,
    rolling_volatility REAL,
    price_momentum REAL,
    target_n0 REAL,
    target_n1 REAL,
    target_n2 REAL,
    target_n3 REAL,
    target_n4 REAL,
    target_n5 REAL,
    future_price_n0 REAL,
    future_price_n1 REAL,
    future_price_n2 REAL,
    future_price_n3 REAL,
    future_price_n4 REAL,
    future_price_n5 REAL,
    PRIMARY KEY (asset, timestamp_utc)
);
