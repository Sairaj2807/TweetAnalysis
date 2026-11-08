# BTC-USD (combined 2019 + 2021-2023 historical backtest, daily candles) — Model Evaluation Report

Generated: 2026-11-08T17:55:31.464911+00:00

## Methodology
- Walk-forward validation with an expanding training window; test folds always come strictly after training data.
- Three feature sets compared: price-only, price+X-sentiment, price+X-sentiment+news.
- Models: XGBoost, LightGBM, evaluated at horizons n+1..n+5.
- Trading metrics (Sharpe, Sortino, Max Drawdown) come from a simple long/short strategy: position direction = sign of predicted price change, realized return = actual price change over that horizon.

## Results

### Model: baseline

| feature_set   |   horizon |       mae |      rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:--------------|----------:|----------:|----------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| buy_and_hold  |         1 |  nan      |  nan      |             nan      |        -0.5222 |         -0.5390 |        -0.5537 |                            nan |
| random_walk   |         1 |  593.0577 |  898.1245 |               0.0000 |         0.0000 |          0.0000 |         0.0000 |                            nan |
| buy_and_hold  |         2 |  nan      |  nan      |             nan      |        -0.5126 |         -0.5372 |        -0.5455 |                            nan |
| random_walk   |         2 |  848.8910 | 1225.1305 |               0.0000 |         0.0000 |          0.0000 |         0.0000 |                            nan |
| buy_and_hold  |         3 |  nan      |  nan      |             nan      |        -0.5950 |         -0.5218 |        -0.5391 |                            nan |
| random_walk   |         3 | 1055.5941 | 1511.3470 |               0.0000 |         0.0000 |          0.0000 |         0.0000 |                            nan |
| buy_and_hold  |         4 |  nan      |  nan      |             nan      |        -0.4992 |         -0.2009 |        -0.5285 |                            nan |
| random_walk   |         4 | 1209.2738 | 1756.8027 |               0.0000 |         0.0000 |          0.0000 |         0.0000 |                            nan |
| buy_and_hold  |         5 |  nan      |  nan      |             nan      |        -0.5190 |         -0.4666 |        -0.5255 |                            nan |
| random_walk   |         5 | 1359.7138 | 1978.1906 |               0.0000 |         0.0000 |          0.0000 |         0.0000 |                            nan |

### Model: lightgbm

| feature_set          |   horizon |       mae |      rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:---------------------|----------:|----------:|----------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only           |         1 |  618.9508 |  920.3789 |               0.4905 |         0.7197 |          1.0928 |        -0.4680 |                         0.0000 |
| price_plus_sentiment |         1 |  611.8050 |  920.5780 |               0.5029 |         0.0581 |          0.2640 |        -0.4928 |                         0.0124 |
| price_only           |         2 |  871.5944 | 1272.3574 |               0.5038 |         0.4449 |          0.8328 |        -0.5159 |                         0.0000 |
| price_plus_sentiment |         2 |  882.9150 | 1284.6256 |               0.4981 |         0.1832 |          0.3003 |        -0.5435 |                        -0.0057 |
| price_only           |         3 | 1083.9414 | 1576.6863 |               0.5019 |        -0.3551 |         -0.4248 |        -0.5598 |                         0.0000 |
| price_plus_sentiment |         3 | 1111.9804 | 1598.2718 |               0.4601 |        -0.4026 |         -0.5582 |        -0.5433 |                        -0.0418 |
| price_only           |         4 | 1242.6602 | 1824.3377 |               0.5143 |         0.1819 |          0.2022 |        -0.4895 |                         0.0000 |
| price_plus_sentiment |         4 | 1298.5145 | 1889.5573 |               0.4876 |         0.4293 |          0.8551 |        -0.4267 |                        -0.0266 |
| price_only           |         5 | 1408.5533 | 2072.7392 |               0.5029 |        -0.2489 |          0.1505 |        -0.5079 |                         0.0000 |
| price_plus_sentiment |         5 | 1476.3122 | 2160.4228 |               0.4819 |        -0.4715 |         -0.4853 |        -0.5674 |                        -0.0209 |

### Model: xgboost

| feature_set          |   horizon |       mae |      rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:---------------------|----------:|----------:|----------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only           |         1 |  616.1478 |  921.9820 |               0.4905 |         0.7756 |          1.1561 |        -0.3916 |                         0.0000 |
| price_plus_sentiment |         1 |  617.0860 |  927.2705 |               0.5095 |         0.0505 |          0.0610 |        -0.4945 |                         0.0190 |
| price_only           |         2 |  878.6175 | 1278.4797 |               0.5038 |         0.7006 |          1.2618 |        -0.4914 |                         0.0000 |
| price_plus_sentiment |         2 |  891.1394 | 1285.4521 |               0.5019 |        -0.3516 |         -0.4075 |        -0.5350 |                        -0.0019 |
| price_only           |         3 | 1084.8305 | 1585.8822 |               0.5086 |        -0.3247 |         -0.4079 |        -0.5405 |                         0.0000 |
| price_plus_sentiment |         3 | 1107.7425 | 1607.8440 |               0.4810 |        -0.1589 |         -0.3364 |        -0.5054 |                        -0.0276 |
| price_only           |         4 | 1254.2499 | 1837.7639 |               0.5143 |         0.4235 |          0.5942 |        -0.4966 |                         0.0000 |
| price_plus_sentiment |         4 | 1293.0585 | 1887.0619 |               0.4952 |         0.2330 |          0.5460 |        -0.4399 |                        -0.0190 |
| price_only           |         5 | 1406.1160 | 2073.4896 |               0.5114 |        -0.1100 |          0.0954 |        -0.4918 |                         0.0000 |
| price_plus_sentiment |         5 | 1470.5801 | 2155.0821 |               0.5010 |        -0.2917 |         -0.2515 |        -0.4938 |                        -0.0105 |

## Key research question

`delta_accuracy_vs_price_only` is the directional-accuracy improvement from adding X sentiment (and news) over price alone. If this is not consistently positive across horizons and folds, there is no evidence yet that sentiment adds forecasting value for this asset/period.