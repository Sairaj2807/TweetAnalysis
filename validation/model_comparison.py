import numpy as np
import pandas as pd

from config.settings import HORIZONS
from features.feature_pipeline import FEATURE_SETS
from models.lightgbm_model import build_lightgbm_model
from models.multi_horizon import predict_multi_horizon, train_multi_horizon
from models.xgboost_model import build_xgboost_model
from validation.leakage_checks import assert_no_overlap
from validation.metrics import PERIODS_PER_YEAR_HOURLY, forecast_metrics, max_drawdown, sharpe_ratio, sortino_ratio, strategy_returns_from_predictions
from validation.walk_forward import walk_forward_windows

MODEL_BUILDERS = {
    "xgboost": build_xgboost_model,
    "lightgbm": build_lightgbm_model,
}


def run_comparison(dataset: pd.DataFrame, n_splits: int = 4, periods_per_year: int = PERIODS_PER_YEAR_HOURLY) -> pd.DataFrame:
    """Runs walk-forward validation across all feature sets, models and horizons.

    Returns a long-format results table: one row per (feature_set, model, horizon, fold)
    with forecast and strategy metrics. This is the evidence for whether sentiment
    (and news) add predictive value over price alone.
    """
    results = []

    for feature_set_name, feature_cols in FEATURE_SETS.items():
        for model_name, builder in MODEL_BUILDERS.items():
            for fold_idx, (train_df, test_df) in enumerate(walk_forward_windows(dataset, n_splits=n_splits)):
                assert_no_overlap(train_df, test_df)

                models = train_multi_horizon(train_df, feature_cols, builder)
                preds = predict_multi_horizon(models, test_df, feature_cols)

                for h in HORIZONS:
                    if h == 0:
                        continue
                    y_true = test_df[f"target_n{h}"].values
                    y_pred = preds[f"n{h}"].values
                    current_price = test_df["price"].values

                    fm = forecast_metrics(y_true, y_pred, current_price)
                    strat_returns = strategy_returns_from_predictions(y_pred, current_price, y_true)

                    results.append(
                        {
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "horizon": h,
                            "fold": fold_idx,
                            **fm,
                            "sharpe_ratio": sharpe_ratio(strat_returns, periods_per_year=periods_per_year),
                            "sortino_ratio": sortino_ratio(strat_returns, periods_per_year=periods_per_year),
                            "max_drawdown": max_drawdown(strat_returns),
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
