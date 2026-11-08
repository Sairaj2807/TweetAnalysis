# BTC-USD (2021-2023 historical backtest, daily candles) — Model Evaluation Report

Generated: 2026-11-08T16:35:06.373971+00:00

## Methodology
- Walk-forward validation with an expanding training window; test folds always come strictly after training data.
- Three feature sets compared: price-only, price+X-sentiment, price+X-sentiment+news.
- Models: XGBoost, LightGBM, evaluated at horizons n+1..n+5.
- Trading metrics (Sharpe, Sortino, Max Drawdown) come from a simple long/short strategy: position direction = sign of predicted price change, realized return = actual price change over that horizon.

## Results

### Model: baseline

| feature_set   |   horizon |       mae |      rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:--------------|----------:|----------:|----------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| buy_and_hold  |         1 |  nan      |  nan      |             nan      |         0.9432 |          2.0387 |        -0.2138 |                            nan |
| random_walk   |         1 |  561.9933 |  801.2688 |               0.0000 |         0.0000 |          0.0000 |         0.0000 |                            nan |
| buy_and_hold  |         2 |  nan      |  nan      |             nan      |         0.8634 |          3.0903 |        -0.2077 |                            nan |
| random_walk   |         2 |  819.5537 | 1135.2492 |               0.0000 |         0.0000 |          0.0000 |         0.0000 |                            nan |
| buy_and_hold  |         3 |  nan      |  nan      |             nan      |         0.5305 |          1.5845 |        -0.2030 |                            nan |
| random_walk   |         3 | 1017.0289 | 1399.1681 |               0.0000 |         0.0000 |          0.0000 |         0.0000 |                            nan |
| buy_and_hold  |         4 |  nan      |  nan      |             nan      |         0.1281 |          0.7410 |        -0.2095 |                            nan |
| random_walk   |         4 | 1208.0009 | 1671.5016 |               0.0000 |         0.0000 |          0.0000 |         0.0000 |                            nan |
| buy_and_hold  |         5 |  nan      |  nan      |             nan      |         0.3092 |          5.6032 |        -0.1682 |                            nan |
| random_walk   |         5 | 1334.9996 | 1843.4603 |               0.0000 |         0.0000 |          0.0000 |         0.0000 |                            nan |

### Model: lightgbm

| feature_set          |   horizon |       mae |      rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:---------------------|----------:|----------:|----------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only           |         1 |  620.3832 |  856.5455 |               0.4696 |        -0.4560 |         -0.0957 |        -0.2566 |                         0.0000 |
| price_plus_sentiment |         1 |  614.3206 |  860.5345 |               0.5372 |         0.0687 |          0.7721 |        -0.2245 |                         0.0676 |
| price_only           |         2 |  900.8330 | 1230.5994 |               0.5169 |        -0.5344 |         -0.7394 |        -0.2361 |                         0.0000 |
| price_plus_sentiment |         2 |  904.3445 | 1249.0381 |               0.4865 |        -1.2600 |         -1.8132 |        -0.2839 |                        -0.0304 |
| price_only           |         3 | 1153.8753 | 1549.9135 |               0.5473 |         1.3077 |          2.7371 |        -0.1899 |                         0.0000 |
| price_plus_sentiment |         3 | 1162.6749 | 1568.6195 |               0.5101 |        -1.0080 |         -1.2209 |        -0.2650 |                        -0.0372 |
| price_only           |         4 | 1334.0658 | 1842.1195 |               0.5304 |         0.9148 |          2.1670 |        -0.2235 |                         0.0000 |
| price_plus_sentiment |         4 | 1370.2059 | 1876.8286 |               0.5068 |         1.2816 |          2.4136 |        -0.1821 |                        -0.0236 |
| price_only           |         5 | 1503.1205 | 2060.8552 |               0.5135 |         1.8100 |          3.0452 |        -0.1063 |                         0.0000 |
| price_plus_sentiment |         5 | 1550.4701 | 2072.7355 |               0.5203 |         1.1972 |          2.7876 |        -0.1606 |                         0.0068 |

### Model: xgboost

| feature_set          |   horizon |       mae |      rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:---------------------|----------:|----------:|----------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only           |         1 |  634.3697 |  879.9879 |               0.5169 |        -0.3734 |         -0.4721 |        -0.2516 |                         0.0000 |
| price_plus_sentiment |         1 |  636.1068 |  889.4687 |               0.5068 |        -1.4378 |         -1.4137 |        -0.2863 |                        -0.0101 |
| price_only           |         2 |  954.4552 | 1289.2881 |               0.4932 |         0.1136 |          0.6084 |        -0.2158 |                         0.0000 |
| price_plus_sentiment |         2 |  951.7061 | 1300.8523 |               0.5101 |        -0.5488 |         -0.2398 |        -0.2496 |                         0.0169 |
| price_only           |         3 | 1175.6215 | 1616.3277 |               0.5574 |         0.0519 |          0.5035 |        -0.2167 |                         0.0000 |
| price_plus_sentiment |         3 | 1192.6258 | 1607.1058 |               0.5135 |        -0.4700 |         -0.5240 |        -0.2221 |                        -0.0439 |
| price_only           |         4 | 1399.4789 | 1926.6067 |               0.5507 |        -0.7288 |         -1.0030 |        -0.2741 |                         0.0000 |
| price_plus_sentiment |         4 | 1381.4845 | 1895.0419 |               0.5270 |         1.1279 |          2.0775 |        -0.1722 |                        -0.0236 |
| price_only           |         5 | 1555.2923 | 2139.4425 |               0.5135 |         1.5340 |          4.2368 |        -0.0982 |                         0.0000 |
| price_plus_sentiment |         5 | 1543.7286 | 2086.9177 |               0.5203 |        -0.0284 |          1.2092 |        -0.1803 |                         0.0068 |

## Key research question

`delta_accuracy_vs_price_only` is the directional-accuracy improvement from adding X sentiment (and news) over price alone. If this is not consistently positive across horizons and folds, there is no evidence yet that sentiment adds forecasting value for this asset/period.