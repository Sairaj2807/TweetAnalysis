# X Sentiment & Price Prediction

Predicts BTC-USD's price at horizons **n+0, n+1, n+2, n+3, n+4, n+5** using
**X (Twitter) sentiment and price only**, with rigorous walk-forward
out-of-sample validation. Built for the CrowdWisdomTrading quant internship
assessment.

> **Security note on credentials:** the Apify token below is a real,
> working credential included here at the repo owner's explicit request for
> reviewer reproducibility. It is publicly visible in this repo and its git
> history — **it should be rotated/revoked in the Apify console once the
> review period is over.** `TAVILY_API_KEY` is kept out of the repo (not yet
> provisioned) and would normally be shared the same way via `.env`.

---

## Interactive dashboard

> ```bash
> pip install -r requirements.txt
> streamlit run streamlit_app.py
> ```
> Opens at **http://localhost:8501**. This is the easiest way to see the
> whole project working without reading any code:
> - **Live Demo tab** — one button runs the entire scrape → ingest → join →
>   train → predict pipeline live, and renders the n+0..n+5 forecast,
>   out-of-sample metrics, and interactive Plotly charts (actual-vs-predicted,
>   sentiment-vs-price) right in the browser.
> - **Historical Backtest tab** — the 2021-2023 evaluation report and chart.
> - **Combined Backtest tab** — the 2014-2023 combined evaluation report and chart.
>
> See [§5e](#5e-interactive-dashboard-optional-nicer-presentation) for details.

---

## 1. What this project does

The assignment: build an ML pipeline that (a) scrapes real X posts about an
asset, (b) ingests recent news, (c) joins posts + news + price into one
database, (d) engineers features from price and sentiment only, and
(e) trains a model that predicts price 0-5 periods ahead, validated
out-of-sample — plus a one-page current sentiment summary and
sentiment-vs-price charts.

This repo does all of that, split into **two pipelines** because of a real
data-availability constraint discovered while building it (see
[Known limitations](#7-known-limitations-read-this)):

| | Purpose | Entry point | Data period | Candle |
|---|---|---|---|---|
| **Live pipeline** | Current sentiment + live forecast | `main.py`, `demo_end_to_end.py` | last ~180 days | hourly |
| **Historical backtest** | The real ML evaluation (bigger sample, walk-forward across market regimes) | `run_historical_backtest.py`, `run_historical_backtest_combined.py` | 2021-2023 (and optionally 2014-2023 combined) | daily |

Both pipelines share the same feature engineering, model, and validation
code — only the data source and candle interval differ.

---

## 2. Architecture

```
ingestion/      Apify (X posts, live), Tavily (news, live), yfinance (price,
                both pipelines), historical_kaggle_loader.py +
                historical_tweets_2019_loader.py (historical tweet datasets)
sentiment/      FinBERT scoring (GPU-accelerated if CUDA available),
                engagement weighting, news sentiment
processing/     cleaning, dedup, UTC/interval alignment, aggregation
database/       SQLite schema joining posts, news, price, features
features/       price/sentiment feature engineering, 3 feature sets, targets
models/         baseline, XGBoost, LightGBM -- one model per horizon (n+1..n+5)
validation/     walk-forward splitting, leakage checks, Sharpe/Sortino/MaxDD
visualization/  sentiment-vs-price overlay, heatmap, prediction chart/matrix
reporting/      one-page sentiment outlook, evaluation report (Markdown)
main.py                          live pipeline (current data)
demo_end_to_end.py                runs all 6 assignment stages in order, prints + saves everything
run_historical_backtest.py        historical pipeline, 2021-2023 (the core ML deliverable)
run_historical_backtest_combined.py  historical pipeline, 2014-2021 + 2021-2023 combined (bigger sample)
streamlit_app.py                  interactive dashboard over all of the above
```

---

## 3. Setup (step by step)

```bash
git clone https://github.com/Sairaj2807/TweetAnalysis.git
cd TweetAnalysis
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and fill in these values (the real Apify token used for this
submission's runs is provided below so the pipeline is reproducible
out of the box):

```ini
APIFY_API_TOKEN=apify_api_6TqSgl2avUaQkurakN3vfWztqt6dAu1zomCE
APIFY_ACTOR_ID=kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest
TAVILY_API_KEY=your_tavily_key_here
DATABASE_URL=sqlite:///data/sentiment_price.db
ASSET_TICKER=BTC-USD
ASSET_CASHTAG=$BTC
ASSET_NAME=Bitcoin
```

- `APIFY_ACTOR_ID` defaults to the actor named above in `config/settings.py`
  even if left blank — see [§4](#4-apify-actor-used--why-this-one) for why
  this specific actor was chosen.
- The token above is pay-per-result against the repo owner's Apify account
  — please be considerate of usage if you run live scrapes against it.
- If `TAVILY_API_KEY` is left blank, `main.py` skips the news step with a
  warning and still runs the rest of the pipeline on price + X sentiment
  alone.

**GPU acceleration (optional but strongly recommended for the historical
backtests):** FinBERT scoring tens of thousands of tweets on CPU can take
close to an hour; on a GPU it takes a few minutes.
`pip install -r requirements.txt` installs CPU-only PyTorch by default. For
CUDA, install a matching build instead, e.g.:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

`sentiment/finbert.py` auto-detects `torch.cuda.is_available()` and uses the
GPU automatically if present — no code changes needed.

---

## 4. Apify actor used + why this one

```
APIFY_ACTOR_ID=kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest
```

We initially tried `apidojo/tweet-scraper`, but its Free-Plan account tier
hard-caps every run at **10 items** regardless of `maxItems` or available
credit (confirmed by testing: raising `maxItems` to 100 still returned
exactly 10). The `kaitoeasyapi` actor has no such cap. It's pay-per-result,
roughly $0.18-0.25 per 1,000 tweets. `ingestion/x_collector.py` searches it
with 9 queries per run: the ticker/cashtag, the asset name, and 7 contextual
bullish/bearish queries (e.g. "bullish", "bearish", "price target", "to the
moon", "crash", "breakout", "sell off") combined with the asset name, each
capped at `maxItems=200`.

**Important, confirmed-by-testing limitation:** this actor (and the other
one we tried) **ignores `since_time`/`until_time` windows set more than a
few days in the past** — requesting tweets from months ago still returns
today's tweets, padded with fabricated placeholder items (`id: "-1"`,
`createdAt: null`) to hit the requested count when too few real matches
exist. `ingestion/x_collector.py` therefore does **not** attempt historical
date-filtering at all, and defensively filters any mock items that slip
through (`_is_mock_item`) — in practice we saw zero mock items once date
filtering was dropped entirely. **Conclusion: Apify is reliable for
current/recent sentiment only, not historical backfill** — hence the
two-pipeline split (§1) and the Kaggle-sourced historical data (§6).

The actor ID and real API token are included above in this README (and
repeated in the submission email) per the assignment's reproducibility
requirement.

---

## 5. How to reproduce every part of the pipeline

Run these in order. Each is independent except where noted.

### 5a. Live pipeline — current sentiment + live forecast
```bash
python main.py
```
Scrapes current real X posts via Apify, current news via Tavily, current
price via yfinance; scores sentiment with FinBERT; builds the feature
dataset; trains + validates; writes:
- `data/reports/sentiment_report.md` — one-page current sentiment outlook
- `data/reports/evaluation_report.md` — MAE/RMSE/direction accuracy/Sharpe/Sortino/MaxDD
- `data/reports/sentiment_vs_price.png`, `data/reports/sentiment_heatmap.png`

### 5b. Full end-to-end demo (the literal assignment deliverable, step by step)
```bash
python demo_end_to_end.py
```
Runs and prints all 6 required stages in order (scrape -> ingest news ->
join in DB -> feature engineering -> train + walk-forward validate ->
produce the n+0..n+5 prediction), reusing `main.py`'s ingestion by default
with cached X posts (no extra Apify spend). Produces, in addition to the
above:
- `data/reports/live_prediction_matrix.csv` / `.png` — the actual n+0..n+5
  forecast, as data and as a rendered visual matrix table
- `data/reports/live_actual_vs_predicted.png` — actual vs. predicted price
  across all 5 horizons on the out-of-sample test window

To force a fresh (paid) Apify run instead of the cache, delete
`data/raw/x_posts_live_cache.csv` first.

### 5c. Historical backtest — 2021-2023 (the core ML evaluation)
```bash
python run_historical_backtest.py
```
Loads a reservoir-sampled subset of the Kaggle `kaushiksuresh147/
bitcoin-tweets` dataset, scores it with FinBERT, fetches matching daily
BTC-USD price, builds features/targets, runs walk-forward validation across
3 feature sets x 2 models x 5 horizons. Produces:
- `data/reports/historical_evaluation_report.md` — the main evaluation deliverable
- `data/reports/historical_sentiment_vs_price.png`
- `data/processed/historical_feature_dataset.csv`

Takes roughly 10-20 minutes on GPU (FinBERT scoring ~46k tweets dominates
the time), much longer on CPU.

### 5d. Combined historical backtest — 2014-2021 + 2021-2023 (bigger sample)
```bash
python run_historical_backtest_combined.py
```
Adds a second, non-overlapping historical window from `data/raw/tweets.csv`
(reservoir-sampled, clipped to 2014-09-17 -> 2021-02-05 — see
`ingestion/historical_tweets_2019_loader.py` for why that exact clip) on top
of the 2021-2023 Kaggle window, giving walk-forward folds more rows and more
market regimes to validate across. Each window's features/targets are built
*separately* before concatenating, so rolling features and `n+1..n+5` shift
targets never bridge the boundary between the two periods (see that script's
docstring for the full reasoning). Produces:
- `data/reports/historical_evaluation_report_combined.md`
- `data/reports/historical_sentiment_vs_price_combined.png`
- `data/processed/historical_feature_dataset_combined.csv`

Takes roughly 30-45 minutes on GPU (scoring ~420k sampled tweets across both
windows dominates the time).

### 5e. Interactive dashboard (optional, nicer presentation)
```bash
pip install streamlit plotly   # already in requirements.txt
streamlit run streamlit_app.py
```
Opens a browser dashboard with three tabs: a "Live Demo" button that runs
the same pipeline as §5b interactively with live status + Plotly charts, and
two tabs that display the already-generated historical/combined reports
from §5c/§5d (run those scripts first — the dashboard reads their saved
output rather than re-running them).

---

## 6. Three feature sets (the core research question)

Every model is trained three times per horizon: **price-only**,
**price + X sentiment**, and **price + X sentiment + news**. Features
engineered from price and sentiment only, per the assignment:

- **Price (time-based):** `return_1`, `return_3`, `return_5` (lagged
  log-returns), `rolling_volatility`, `price_momentum`
  (`features/price_features.py`)
- **Sentiment (time-based):** `sentiment_mean`, `sentiment_weighted`
  (engagement-weighted), `post_count`, `positive_ratio`, `negative_ratio`,
  `engagement_total`, `sentiment_change`, `sentiment_lag_1`
  (`features/sentiment_features.py`, `features/feature_pipeline.py`)
- **Time horizon:** every row carries 6 target columns, `target_n0..target_n5`
  (`HORIZONS` in `config/settings.py`), one per forecast horizon

The evaluation report's `delta_accuracy_vs_price_only` column is the
headline result — it answers whether sentiment (and news) actually add
forecasting value, rather than just producing an attractive-looking chart.

---

## 7. Known limitations (read this)

- **Apify cannot reliably backfill months of X history — confirmed by
  testing, not assumed.** See [§4](#4-apify-actor-used--why-this-one) for
  the detail. Conclusion: Apify is used for **current/recent** sentiment
  only (`main.py`), never for historical backfill.
- **Historical training data comes from two pre-existing datasets instead:**
  - `kaushiksuresh147/bitcoin-tweets` (Kaggle, ~5.6M tweets, 2021-02-05 to
    2023-01-09, no per-tweet engagement counts, only follower counts).
  - `data/raw/tweets.csv` (semicolon-delimited, has real per-tweet
    likes/retweets/replies but no follower counts) — this file turned out
    to span **2011-2019**, not the single 6-month 2019 window it first
    appeared to be from a head/tail check; it's clipped to
    **2014-09-17 -> 2021-02-05** in `historical_tweets_2019_loader.py`
    because 2014-09-17 is yfinance's earliest available BTC-USD daily bar
    (confirmed by testing) and 2021-02-05 is where the Kaggle window
    begins, keeping the two windows non-overlapping.
  - Both are reservoir-sampled (per calendar day) to keep FinBERT scoring
    tractable, since scoring either dataset in full would take hours even
    on GPU.
  - This means the historical backtests and the live sentiment report cover
    **different time periods** from "now" — two separate, honestly-scoped
    halves of the assignment, not one continuous dataset. Stored under
    distinct `asset` keys (`BTC-USD_HIST`, `BTC-USD_HIST_COMBINED`) so they
    never collide with live `BTC-USD` rows.
- **Historical backtests use daily candles, not hourly.** yfinance's hourly
  data only covers the last ~730 days, which doesn't reach back to 2014 or
  2021; daily bars have no such limit. `n+1..n+5` in the historical
  evaluation reports means **days** ahead, not hours — Sharpe/Sortino are
  annualized with 365 periods/year for these reports vs. the live
  pipeline's hourly annualization (`validation/metrics.py:
  PERIODS_PER_YEAR_HOURLY` vs. `DAILY_PERIODS_PER_YEAR` in the backtest
  scripts).
- **Tavily news is current-only.** Tavily's search API returns what's
  indexed now; it cannot be queried "as of" a past timestamp. News sentiment
  is therefore only populated for the current one-page sentiment report;
  `news_sentiment` is 0 for both the live dataset's early history and the
  entire historical backtest periods.
- **The 2014-2021 dataset has no per-tweet follower counts** (only
  likes/retweets/replies); the **2021-2023 Kaggle dataset has no per-tweet
  engagement** (only follower counts). Engagement weighting therefore uses
  whichever signal is available per source, not a perfectly uniform
  methodology across the combined backtest.
- **Walk-forward windows are a simple expanding-window implementation**
  (`validation/walk_forward.py`), not purged/embargoed cross-validation.
- Sharpe/Sortino/Max Drawdown are computed from a simple **long/short
  directional strategy** (position = sign of predicted price change), not a
  production trading system with transaction costs or slippage.
- **The evaluation results are genuinely mixed** — sentiment improves
  directional accuracy at some horizons/models and hurts at others, and
  out-of-sample direction accuracy sits close to the 50% coin-flip baseline
  overall. This is reported as-is; we did not cherry-pick a feature set or
  horizon to present a cleaner "sentiment helps" story than the data
  supports. See the evaluation reports themselves for the full numbers.

---

## 8. Sample saved model

`saved_models/sample_xgboost_multi_horizon.pkl` is a standalone sample
artifact: all 5 horizon models (n+1..n+5), trained once on
`data/processed/historical_feature_dataset_combined.csv` (2,633 rows) and
saved with `joblib`, included so a reviewer can inspect/load a concrete
trained model without running the pipeline first.

**It is not loaded or referenced by any pipeline script.** `main.py`,
`demo_end_to_end.py`, `run_historical_backtest*.py`, and `streamlit_app.py`
all retrain fresh in memory every run, by design — this file is purely a
point-in-time sample, not a production artifact the app depends on.

To load and use it:
```python
import joblib
artifact = joblib.load("saved_models/sample_xgboost_multi_horizon.pkl")
models, feature_cols = artifact["models"], artifact["feature_cols"]
# models[1] is the n+1 XGBoost model, models[2] is n+2, etc.
```

## 9. Prototype

`prototype/` contains the original throwaway proof-of-concept (yfinance +
mock tweet text + a single XGBoost model) used to validate the pipeline
shape before building the real structure above. Not part of the submission.
