# BTC-USD (2021-2023 historical backtest, daily candles) — Model Evaluation Report

Generated: 2026-11-08T07:58:29.859858+00:00

## Methodology
- Walk-forward validation with an expanding training window; test folds always come strictly after training data.
- Three feature sets compared: price-only, price+X-sentiment, price+X-sentiment+news.
- Models: XGBoost, LightGBM, evaluated at horizons n+1..n+5.
- Trading metrics (Sharpe, Sortino, Max Drawdown) come from a simple long/short strategy: position direction = sign of predicted price change, realized return = actual price change over that horizon.

## Results

### Model: lightgbm

| feature_set                    |   horizon |       mae |      rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:-------------------------------|----------:|----------:|----------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only                     |         1 | 2829.5425 | 3255.0854 |               0.4764 |         0.0044 |          0.1197 |        -0.2273 |                         0.0000 |
| price_plus_sentiment           |         1 | 2860.9562 | 3284.3535 |               0.4932 |         0.2928 |          0.4535 |        -0.2403 |                         0.0169 |
| price_plus_sentiment_plus_news |         1 | 2860.9562 | 3284.3535 |               0.4932 |         0.2928 |          0.4535 |        -0.2403 |                         0.0169 |
| price_only                     |         2 | 3217.6460 | 3635.7594 |               0.5203 |        -0.2409 |         -0.4495 |        -0.3836 |                         0.0000 |
| price_plus_sentiment           |         2 | 3125.0893 | 3554.1580 |               0.5270 |         0.5349 |          0.6872 |        -0.3652 |                         0.0068 |
| price_plus_sentiment_plus_news |         2 | 3125.0893 | 3554.1580 |               0.5270 |         0.5349 |          0.6872 |        -0.3652 |                         0.0068 |
| price_only                     |         3 | 3388.5668 | 3834.4835 |               0.5236 |         0.0507 |         -0.1348 |        -0.4437 |                         0.0000 |
| price_plus_sentiment           |         3 | 3394.3868 | 3852.9400 |               0.5405 |         0.7872 |          1.3199 |        -0.4718 |                         0.0169 |
| price_plus_sentiment_plus_news |         3 | 3394.3868 | 3852.9400 |               0.5405 |         0.7872 |          1.3199 |        -0.4718 |                         0.0169 |
| price_only                     |         4 | 3628.4923 | 4115.3363 |               0.5709 |         1.9399 |          3.4083 |        -0.4416 |                         0.0000 |
| price_plus_sentiment           |         4 | 3576.8834 | 4032.7377 |               0.5608 |         1.2540 |          2.3237 |        -0.5005 |                        -0.0101 |
| price_plus_sentiment_plus_news |         4 | 3576.8834 | 4032.7377 |               0.5608 |         1.2540 |          2.3237 |        -0.5005 |                        -0.0101 |
| price_only                     |         5 | 3790.2303 | 4303.2690 |               0.5608 |         1.3575 |          2.2141 |        -0.4890 |                         0.0000 |
| price_plus_sentiment           |         5 | 3568.5684 | 4047.6813 |               0.5507 |         1.2638 |          2.4899 |        -0.5683 |                        -0.0101 |
| price_plus_sentiment_plus_news |         5 | 3568.5684 | 4047.6813 |               0.5507 |         1.2638 |          2.4899 |        -0.5683 |                        -0.0101 |

### Model: xgboost

| feature_set                    |   horizon |       mae |      rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:-------------------------------|----------:|----------:|----------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only                     |         1 | 3075.3702 | 3486.6933 |               0.4899 |         0.7148 |          1.1126 |        -0.2173 |                         0.0000 |
| price_plus_sentiment           |         1 | 3131.3667 | 3560.7595 |               0.4865 |         0.9311 |          1.5466 |        -0.2038 |                        -0.0034 |
| price_plus_sentiment_plus_news |         1 | 2911.8208 | 3363.9148 |               0.4932 |         0.7975 |          1.2807 |        -0.2351 |                         0.0034 |
| price_only                     |         2 | 3352.8320 | 3780.4181 |               0.5439 |         1.4519 |          2.3140 |        -0.3339 |                         0.0000 |
| price_plus_sentiment           |         2 | 3480.2991 | 3959.7051 |               0.5236 |         0.5185 |          0.6747 |        -0.3562 |                        -0.0203 |
| price_plus_sentiment_plus_news |         2 | 3268.9122 | 3703.8804 |               0.5338 |         1.0811 |          1.5645 |        -0.3226 |                        -0.0101 |
| price_only                     |         3 | 3650.0817 | 4094.9506 |               0.5270 |         0.4179 |          0.7476 |        -0.4622 |                         0.0000 |
| price_plus_sentiment           |         3 | 3717.0589 | 4196.1633 |               0.5270 |        -0.1241 |         -0.3007 |        -0.4519 |                         0.0000 |
| price_plus_sentiment_plus_news |         3 | 3506.0068 | 3973.5864 |               0.5270 |         0.9853 |          1.4321 |        -0.3953 |                         0.0000 |
| price_only                     |         4 | 3923.7713 | 4383.9150 |               0.5608 |         1.0706 |          2.2819 |        -0.4686 |                         0.0000 |
| price_plus_sentiment           |         4 | 3870.5436 | 4365.1577 |               0.5507 |         1.0565 |          1.7296 |        -0.4581 |                        -0.0101 |
| price_plus_sentiment_plus_news |         4 | 3681.2178 | 4159.8624 |               0.5574 |         1.0453 |          1.6865 |        -0.4847 |                        -0.0034 |
| price_only                     |         5 | 3901.1157 | 4405.6932 |               0.5608 |         1.7389 |          4.2413 |        -0.4773 |                         0.0000 |
| price_plus_sentiment           |         5 | 3834.1347 | 4326.3420 |               0.5405 |         0.9189 |          1.7197 |        -0.5293 |                        -0.0203 |
| price_plus_sentiment_plus_news |         5 | 3659.3750 | 4134.3418 |               0.5676 |         2.1288 |          4.7092 |        -0.5252 |                         0.0068 |

## Key research question

`delta_accuracy_vs_price_only` is the directional-accuracy improvement from adding X sentiment (and news) over price alone. If this is not consistently positive across horizons and folds, there is no evidence yet that sentiment adds forecasting value for this asset/period.