# X Sentiment & Price Prediction

Predicts an asset's price at horizons n+0..n+5 using X (Twitter) sentiment,
recent news sentiment, and price alone, with walk-forward out-of-sample
validation. Built for the CrowdWisdomTrading quant internship assessment.

## Architecture

```
ingestion/      Apify (X posts), Tavily (news), yfinance (price bars)
sentiment/      FinBERT scoring, engagement weighting, news sentiment
processing/     cleaning, dedup, UTC/interval alignment, aggregation
database/       SQLite schema joining posts, news, price, features
features/       price/sentiment feature engineering, 3 feature sets, targets
models/         baseline, XGBoost, LightGBM, one model per horizon (n+1..n+5)
validation/     walk-forward splitting, leakage checks, Sharpe/Sortino/MaxDD
visualization/  sentiment-vs-price overlay, heatmap, prediction charts/matrix
reporting/      one-page sentiment outlook, evaluation report (Markdown)
main.py         runs the full pipeline end to end
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in APIFY_API_TOKEN, APIFY_ACTOR_ID, TAVILY_API_KEY in .env
python main.py
```

If Apify/Tavily keys are not set, `main.py` skips those steps with a warning
and still runs the full pipeline on price data alone -- useful for verifying
the plumbing before API access is configured.

## Apify actor used

Set `APIFY_ACTOR_ID` in `.env` to an X/Twitter search-scraper actor from the
Apify Store (searched by ticker/cashtag, asset name, and contextual bullish/
bearish queries -- see `ingestion/x_collector.py`). The actor ID and a run log
are included in the submission email per the assignment's reproducibility
requirement.

## Three feature sets (the core research question)

Every model is trained three times per horizon: **price-only**,
**price + X sentiment**, and **price + X sentiment + news**. The evaluation
report's `delta_accuracy_vs_price_only` column is the headline result -- it
answers whether sentiment (and news) actually add forecasting value, rather
than just producing an attractive-looking chart.

## Known limitations

- **Tavily news is current-only.** Tavily's search API returns what's indexed
  now; it cannot be queried "as of" a past timestamp. News sentiment is
  therefore most reliable for the current one-page sentiment report. For
  historical training rows, `news_sentiment` is 0 unless a timestamped news
  archive is wired in separately.
- **180-day X history depends on the chosen Apify actor's actual coverage.**
  X's own search limits mean a successful scrape does not guarantee every
  historical post was retrieved -- treat `post_count` as a lower bound.
- **Walk-forward windows are a simple expanding-window implementation**
  (`validation/walk_forward.py`), not purged/embargoed cross-validation. For a
  1-hour candle interval this is adequate since targets don't overlap folds
  by more than a few hours; revisit if the candle interval changes.
- Sharpe/Sortino/Max Drawdown are computed from a simple **long/short
  directional strategy** (position = sign of predicted price change), not a
  production trading system with transaction costs or slippage.

## Prototype

`prototype/` contains the original throwaway proof-of-concept (yfinance +
mock tweet text + single XGBoost model) used to validate the pipeline shape
before building the real structure above. Not part of the submission.
