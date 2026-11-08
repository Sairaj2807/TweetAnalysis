import numpy as np
import pandas as pd

from config.settings import HORIZONS
from features.feature_pipeline import FEATURE_SETS
from models.baseline import persistence_predict
from models.lightgbm_model import build_lightgbm_model
from models.multi_horizon import predict_multi_horizon_price, train_multi_horizon
from models.xgboost_model import build_xgboost_model
from validation.leakage_checks import assert_no_overlap
from validation.metrics import PERIODS_PER_YEAR_HOURLY, forecast_metrics, max_drawdown, sharpe_ratio, sortino_ratio, strategy_returns_from_predictions
from validation.walk_forward import walk_forward_windows

MODEL_BUILDERS = {
    "xgboost": build_xgboost_model,
    "lightgbm": build_lightgbm_model,
}

DEFAULT_COST_BPS = 10.0  # conservative round-trip transaction cost estimate


def _strategy_metrics(y_true_price, y_pred_price, current_price, h, periods_per_year, cost_bps):
    """Forecast metrics plus Sharpe/Sortino/MaxDD computed on NON-overlapping h-day
    blocks (every h-th row), annualized with periods_per_year/h. Without this, each
    n+5 "return" is really a 5-day return but gets annualized with sqrt(periods_per_year)
    as if it were a 1-day observation, inflating Sharpe by roughly sqrt(h) and
    double/triple/quadruple/quintuple-counting the same underlying days across folds."""
    fm = forecast_metrics(y_true_price, y_pred_price, current_price)
    strat_returns = strategy_returns_from_predictions(y_pred_price, current_price, y_true_price, cost_bps=cost_bps)
    nonoverlap_returns = strat_returns[::h] if h > 0 else strat_returns
    effective_periods_per_year = max(periods_per_year / h, 1e-9)
    return {
        **fm,
        "sharpe_ratio": sharpe_ratio(nonoverlap_returns, periods_per_year=effective_periods_per_year),
        "sortino_ratio": sortino_ratio(nonoverlap_returns, periods_per_year=effective_periods_per_year),
        "max_drawdown": max_drawdown(nonoverlap_returns),
    }


def run_comparison(
    dataset: pd.DataFrame,
    n_splits: int = 4,
    periods_per_year: int = PERIODS_PER_YEAR_HOURLY,
    feature_sets: dict = None,
    purge: int = None,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    """Runs walk-forward validation across all feature sets, models and horizons,
    plus two model-free reference rows: random_walk (persistence -- predicted price
    = current price) and buy_and_hold (always long). Without these two, MAE and
    Sharpe numbers have no baseline to be judged against.

    `purge` defaults to max(HORIZONS) so no training row's shift(-h) target can reach
    into any test window (see validation/walk_forward.py). `cost_bps` defaults to a
    conservative 10bp round-trip cost so Sharpe/MaxDD reflect paying for trades, not
    a frictionless strategy. `feature_sets` defaults to the full FEATURE_SETS dict but
    callers (e.g. the historical backtests) can pass a trimmed one -- see
    features/feature_pipeline.py: HISTORICAL_FEATURE_SETS / COMBINED_HISTORICAL_FEATURE_SETS.

    Returns a long-format results table: one row per (feature_set, model, horizon, fold)
    with forecast and strategy metrics. This is the evidence for whether sentiment
    (and news) add predictive value over price alone -- and now, over doing nothing
    (random_walk) or just holding (buy_and_hold).
    """
    feature_sets = feature_sets if feature_sets is not None else FEATURE_SETS
    if purge is None:
        purge = max((h for h in HORIZONS if h > 0), default=0)

    results = []

    for feature_set_name, feature_cols in feature_sets.items():
        for model_name, builder in MODEL_BUILDERS.items():
            for fold_idx, (train_df, test_df) in enumerate(walk_forward_windows(dataset, n_splits=n_splits, purge=purge)):
                assert_no_overlap(train_df, test_df)

                models = train_multi_horizon(train_df, feature_cols, builder)
                preds = predict_multi_horizon_price(models, test_df, feature_cols)

                for h in HORIZONS:
                    if h == 0:
                        continue
                    y_true = test_df[f"future_price_n{h}"].values
                    y_pred = preds[f"n{h}"].values
                    current_price = test_df["price"].values

                    row = _strategy_metrics(y_true, y_pred, current_price, h, periods_per_year, cost_bps)
                    results.append({"feature_set": feature_set_name, "model": model_name, "horizon": h, "fold": fold_idx, **row})

    # Model-free baselines, evaluated on the same folds/purge so they're directly
    # comparable to the trained models above.
    for fold_idx, (train_df, test_df) in enumerate(walk_forward_windows(dataset, n_splits=n_splits, purge=purge)):
        for h in HORIZONS:
            if h == 0:
                continue
            current_price = test_df["price"].values
            y_true = test_df[f"future_price_n{h}"].values

            # random_walk: predicted price = current price (zero predicted change).
            # Direction accuracy is degenerate here (predicted direction is always
            # exactly 0, so it essentially never matches a nonzero actual direction)
            # -- MAE is the meaningful comparison for this baseline, not direction accuracy.
            rw_pred = persistence_predict(test_df, h)
            rw_row = _strategy_metrics(y_true, rw_pred, current_price, h, periods_per_year, cost_bps=0.0)
            results.append({"feature_set": "random_walk", "model": "baseline", "horizon": h, "fold": fold_idx, **rw_row})

            # buy_and_hold: always long, never flips position, so there's no
            # transaction cost to apply and no "prediction" to score for MAE/direction.
            bh_returns = (y_true - current_price) / current_price
            nonoverlap = bh_returns[::h]
            effective_ppy = max(periods_per_year / h, 1e-9)
            results.append(
                {
                    "feature_set": "buy_and_hold",
                    "model": "baseline",
                    "horizon": h,
                    "fold": fold_idx,
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "r2": float("nan"),
                    "direction_accuracy": float("nan"),
                    "pearson_correlation": float("nan"),
                    "sharpe_ratio": sharpe_ratio(nonoverlap, periods_per_year=effective_ppy),
                    "sortino_ratio": sortino_ratio(nonoverlap, periods_per_year=effective_ppy),
                    "max_drawdown": max_drawdown(nonoverlap),
                }
            )

    return pd.DataFrame(results)


def summarize_by_feature_set_and_horizon(results: pd.DataFrame) -> pd.DataFrame:
    """Averages fold-level results, and computes delta accuracy vs. price_only for each model/horizon."""
    summary = (
        results.groupby(["model", "feature_set", "horizon"])
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            direction_accuracy=("direction_accuracy", "mean"),
            r2=("r2", "mean"),
            pearson_correlation=("pearson_correlation", "mean"),
            sharpe_ratio=("sharpe_ratio", "mean"),
            sortino_ratio=("sortino_ratio", "mean"),
            max_drawdown=("max_drawdown", "mean"),
        )
        .reset_index()
    )

    baseline = summary[summary["feature_set"] == "price_only"][["model", "horizon", "direction_accuracy"]]
    baseline = baseline.rename(columns={"direction_accuracy": "price_only_accuracy"})
    summary = summary.merge(baseline, on=["model", "horizon"], how="left")
    summary["delta_accuracy_vs_price_only"] = summary["direction_accuracy"] - summary["price_only_accuracy"]
    return summary
