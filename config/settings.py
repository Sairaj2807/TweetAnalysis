import os
from dotenv import load_dotenv

load_dotenv()

APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
APIFY_ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/sentiment_price.db")

ASSET_TICKER = os.getenv("ASSET_TICKER", "BTC-USD")
ASSET_CASHTAG = os.getenv("ASSET_CASHTAG", "$BTC")
ASSET_NAME = os.getenv("ASSET_NAME", "Bitcoin")

HISTORY_DAYS = 180
CANDLE_INTERVAL = "1h"
HORIZONS = [0, 1, 2, 3, 4, 5]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
