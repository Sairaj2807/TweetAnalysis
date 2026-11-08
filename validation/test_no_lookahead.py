"""Regression test proving features/feature_pipeline.py never looks into the future:
changing a later day's price must not change any earlier row's engineered features.

This formalizes an invariant that was previously only an informal claim (the code
only ever uses .shift()/.rolling() looking backward) -- see Gilad's review feedback
(Q8): "write a test that changes the close of day t+1 and checks that no feature of
row t changes."

Run with: python -m pytest validation/test_no_lookahead.py -v
"""
import numpy as np
import pandas as pd

from features.feature_pipeline import PRICE_ONLY_FEATURES, SENTIMENT_FEATURES, build_features

N_ROWS = 30
T = 20  # the row whose features must stay fixed
T_PLUS_1 = T + 1  # the row whose price we perturb


def _make_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=N_ROWS, freq="D", tz="UTC")
    price = 20000 + np.cumsum(rng.normal(0, 200, N_ROWS))
    return pd.DataFrame(
        {
            "timestamp_utc": dates,
            "price": price,
            "sentiment_mean": rng.normal(0, 0.3, N_ROWS),
            "sentiment_weighted": rng.normal(0, 0.3, N_ROWS),
            "positive_ratio": rng.uniform(0, 1, N_ROWS),
            "negative_ratio": rng.uniform(0, 1, N_ROWS),
            "post_count": rng.integers(1, 50, N_ROWS),
            "engagement_total": rng.uniform(0, 1000, N_ROWS),
            "news_sentiment": rng.normal(0, 0.3, N_ROWS),
        }
    )


def test_features_unaffected_by_future_price_change():
    baseline = build_features(_make_frame())

    perturbed_input = _make_frame()
    perturbed_input.loc[T_PLUS_1, "price"] *= 1.5  # drastic change to tomorrow's close
    perturbed = build_features(perturbed_input)

    feature_cols = PRICE_ONLY_FEATURES + SENTIMENT_FEATURES
    pd.testing.assert_series_equal(
        baseline.loc[T, feature_cols], perturbed.loc[T, feature_cols], check_names=False
    )


if __name__ == "__main__":
    test_features_unaffected_by_future_price_change()
    print("PASSED: row t's features are unaffected by a drastic change to row t+1's price.")
