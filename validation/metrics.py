import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PERIODS_PER_YEAR_HOURLY = 24 * 365


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray, current_price: np.ndarray) -> dict:
    """Standard forecast-quality metrics for one horizon."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    actual_dir = np.sign(y_true - current_price)
    pred_dir = np.sign(y_pred - current_price)
    direction_accuracy = float((actual_dir == pred_dir).mean())

    actual_return = (y_true - current_price) / current_price
    pred_return = (y_pred - current_price) / current_price
    corr = pearsonr(actual_return, pred_return)[0] if len(actual_return) > 1 else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "direction_accuracy": direction_accuracy,
        "pearson_correlation": corr,
    }


def strategy_returns_from_predictions(
    y_pred: np.ndarray, current_price: np.ndarray, actual_price: np.ndarray, cost_bps: float = 0.0
) -> np.ndarray:
    """Converts predictions into a simple long/short directional strategy's realized returns.

    Signal: go long if predicted price > current price, short if lower. Position
    is held for one horizon period; realized return is the actual price change
    over that period, signed by the predicted direction.

    `cost_bps`: round-trip transaction cost in basis points charged whenever the
    position changes from the previous period (0 by default for backward
    compatibility, but callers should pass a nonzero value for a realistic net
    return -- the unadjusted version books gross returns with no cost at all).
    """
    predicted_direction = np.sign(y_pred - current_price)
    actual_return = (actual_price - current_price) / current_price
    gross = predicted_direction * actual_return
    if cost_bps > 0:
        position_change = np.abs(np.diff(np.concatenate([[0], predicted_direction])))
        gross = gross - position_change * (cost_bps / 10000)
    return gross


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = PERIODS_PER_YEAR_HOURLY, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / periods_per_year
    if excess.std(ddof=1) == 0:
        return 0.0
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = PERIODS_PER_YEAR_HOURLY, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())
