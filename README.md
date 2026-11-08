# X Sentiment & Price Prediction

Predicts an asset's price at horizons n+0..n+5 using X (Twitter) sentiment,
news sentiment, and price alone, with walk-forward out-of-sample validation.
Built for the CrowdWisdomTrading quant internship assessment.

## Two pipelines

This project has two entry points, because real-world data availability split
the "current" and "historical" halves of the assignment (see Known Limitations):

- **`main.py`** -- live pipeline. Scrapes current real X posts via Apify,
  current price via yfinance, scores sentiment, writes the one-page current
  sentiment outlook. This is the live/current half of the assignment.
- **`run_historical_backtest.py`** -- the real ML deliverable. Uses a
  pre-existing historical tweets dataset (Kaggle) + matching historical daily
  BTC-USD price to build the training dataset, run walk-forward validation
  across 3 feature sets x 2 models x 5 horizons, and produce the evaluation
  report with MAE/RMSE/direction accuracy/Sharpe/Sortino/Max Drawdown --
  `data/reports/historical_evaluation_report.md`.

## Architecture

```
ingestion/      Apify (X posts), Tavily (news), yfinance (price bars),
                historical_kaggle_loader.py (historical tweets dataset)
sentiment/      FinBERT scoring (GPU-accelerated if CUDA available),
                engagement weighting, news sentiment
processing/     cleaning, dedup, UTC/interval alignment, aggregation
database/       SQLite schema joining posts, news, price, features
features/       price/sentiment feature engineering, 3 feature sets, targets
models/         baseline, XGBoost, LightGBM, one model per horizon (n+1..n+5)
validation/     walk-forward splitting, leakage checks, Sharpe/Sortino/MaxDD
visualization/  sentiment-vs-price overlay, heatmap, prediction charts/matrix
reporting/      one-page sentiment outlook, evaluation report (Markdown)
main.py                     live pipeline (current data)
run_historical_backtest.py  historical pipeline (the ML deliverable)
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

**GPU acceleration (optional but strongly recommended):** FinBERT scoring
~45k+ historical tweets on CPU takes close to an hour; on a GPU it takes a
few minutes. `pip install -r requirements.txt` installs CPU-only PyTorch by
default. For CUDA, install a matching build instead, e.g.:
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```
`sentiment/finbert.py` auto-detects `torch.cuda.is_available()` and uses the
GPU automatically if present -- no code changes needed.

## Apify actor used

`APIFY_ACTOR_ID=kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest`
(see `.env.example`). We initially tried `apidojo/tweet-scraper`, but its
Free-Plan account tier hard-caps every run at 10 items regardless of
`maxItems` or available credit (confirmed by testing: raising `maxItems` to
100 still returned exactly 10). The kaitoeasyapi actor has no such cap.
Pay-per-result, ~$0.18-0.25 per 1,000 tweets. Searched by ticker/cashtag,
asset name, and contextual bullish/bearish queries -- see
`ingestion/x_collector.py`. The actor ID, token, and a run log are included
in the submission email per the assignment's reproducibility requirement.

## Three feature sets (the core research question)

Every model is trained three times per horizon: **price-only**,
**price + X sentiment**, and **price + X sentiment + news**. The evaluation
report's `delta_accuracy_vs_price_only` column is the headline result -- it
answers whether sentiment (and news) actually add forecasting value, rather
than just producing an attractive-looking chart.

## Known limitations

- **Apify cannot reliably backfill months of X history -- confirmed by
  testing, not assumed.** Both actors we tried (`apidojo/tweet-scraper` and
  `kaitoeasyapi/...`) ignore `since_time`/`until_time` windows set more than a
  few days in the past: requesting tweets from May 2026 still returned
  tweets from today, padded with fabricated placeholder items (`id: "-1"`,
  `createdAt: null`) to hit the requested count when too few real matches
  existed. `ingestion/x_collector.py` defensively filters these out
  (`_is_mock_item`), and in practice we saw **zero** mock items once date
  filtering was dropped entirely. Conclusion: Apify is reliable for
  **current/recent** sentiment (used by `main.py`), not historical backfill.
- **Historical training data comes from a pre-existing Kaggle dataset
  instead** (`kaushiksuresh147/bitcoin-tweets`, ~5.6M tweets, 2021-02-05 to
  2023-01-09, no per-tweet engagement counts). `run_historical_backtest.py`
  reservoir-samples ~150-200 real tweets/day (unbiased across the full
  period) to keep FinBERT scoring tractable, since scoring all 5.6M rows
  would take hours even on GPU. This means the historical backtest and the
  live sentiment report cover **different time periods** (2021-2023 vs. now)
  -- they are two separate, honestly-scoped halves of the assignment, not one
  continuous dataset. Stored under a distinct `asset` key (`BTC-USD_HIST`) so
  it never collides with live `BTC-USD` rows.
- **Historical backtest uses daily candles, not hourly.** yfinance's hourly
  data only covers the last ~730 days, which doesn't reach back to 2021;
  daily bars have no such limit. `n+1..n+5` in the historical evaluation
  report means **days** ahead, not hours -- Sharpe/Sortino are annualized
  with 365 periods/year for this report vs. the live pipeline's hourly
  annualization (see `validation/metrics.py: PERIODS_PER_YEAR_HOURLY` and
  `run_historical_backtest.py: DAILY_PERIODS_PER_YEAR`).
- **Tavily news is current-only.** Tavily's search API returns what's indexed
  now; it cannot be queried "as of" a past timestamp. News sentiment is
  therefore only populated for the current one-page sentiment report;
  `news_sentiment` is 0 for both the live dataset's early history and the
  entire 2021-2023 historical backtest.
- The Kaggle dataset has **no per-tweet like/retweet/reply counts** (only
  user-level follower counts), so `engagement_weighted` sentiment for the
  historical backtest falls back to follower-count weighting only, unlike
  the live pipeline which has full engagement metrics.
- **Walk-forward windows are a simple expanding-window implementation**
  (`validation/walk_forward.py`), not purged/embargoed cross-validation.
- Sharpe/Sortino/Max Drawdown are computed from a simple **long/short
  directional strategy** (position = sign of predicted price change), not a
  production trading system with transaction costs or slippage.
- The historical evaluation report's results are **genuinely mixed** --
  sentiment improves directional accuracy at some horizons/models and hurts
  at others. This is reported as-is; we did not cherry-pick a feature set or
  horizon to present a cleaner "sentiment helps" story than the data
  supports.

## Prototype

`prototype/` contains the original throwaway proof-of-concept (yfinance +
mock tweet text + single XGBoost model) used to validate the pipeline shape
before building the real structure above. Not part of the submission.
