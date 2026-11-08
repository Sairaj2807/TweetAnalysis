# BTC-USD (combined 2019 + 2021-2023 historical backtest, daily candles) — Model Evaluation Report

Generated: 2026-11-08T12:17:20.130344+00:00

## Methodology
- Walk-forward validation with an expanding training window; test folds always come strictly after training data.
- Three feature sets compared: price-only, price+X-sentiment, price+X-sentiment+news.
- Models: XGBoost, LightGBM, evaluated at horizons n+1..n+5.
- Trading metrics (Sharpe, Sortino, Max Drawdown) come from a simple long/short strategy: position direction = sign of predicted price change, realized return = actual price change over that horizon.

## Results

### Model: lightgbm

| feature_set                    |   horizon |       mae |      rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:-------------------------------|----------:|----------:|----------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only                     |         1 | 3087.4519 | 5376.5023 |               0.4971 |         0.0418 |          0.1329 |        -0.5117 |                         0.0000 |
| price_plus_sentiment           |         1 | 4929.0971 | 7496.0203 |               0.4905 |        -0.0869 |         -0.0553 |        -0.5170 |                        -0.0067 |
| price_plus_sentiment_plus_news |         1 | 4929.0971 | 7496.0203 |               0.4905 |        -0.0869 |         -0.0553 |        -0.5170 |                        -0.0067 |
| price_only                     |         2 | 3195.6556 | 5449.4037 |               0.5095 |         0.4734 |          0.7569 |        -0.6160 |                         0.0000 |
| price_plus_sentiment           |         2 | 5076.2190 | 7608.5625 |               0.5105 |         0.2346 |          0.4318 |        -0.6310 |                         0.0010 |
| price_plus_sentiment_plus_news |         2 | 5076.2190 | 7608.5625 |               0.5105 |         0.2346 |          0.4318 |        -0.6310 |                         0.0010 |
| price_only                     |         3 | 3443.2737 | 5737.0275 |               0.5124 |         0.3366 |          0.5549 |        -0.7390 |                         0.0000 |
| price_plus_sentiment           |         3 | 5206.8956 | 7755.0566 |               0.5247 |         0.5260 |          1.0263 |        -0.7139 |                         0.0124 |
| price_plus_sentiment_plus_news |         3 | 5206.8956 | 7755.0566 |               0.5247 |         0.5260 |          1.0263 |        -0.7139 |                         0.0124 |
| price_only                     |         4 | 3791.9667 | 6151.6378 |               0.5143 |        -0.0225 |          0.0667 |        -0.8230 |                         0.0000 |
| price_plus_sentiment           |         4 | 5690.9844 | 8219.7321 |               0.5029 |         0.1713 |          0.3991 |        -0.8007 |                        -0.0114 |
| price_plus_sentiment_plus_news |         4 | 5690.9844 | 8219.7321 |               0.5029 |         0.1713 |          0.3991 |        -0.8007 |                        -0.0114 |
| price_only                     |         5 | 4127.0128 | 6517.6990 |               0.5200 |         0.1425 |          0.3235 |        -0.8606 |                         0.0000 |
| price_plus_sentiment           |         5 | 5979.8531 | 8465.7198 |               0.5124 |         0.1376 |          0.4778 |        -0.8494 |                        -0.0076 |
| price_plus_sentiment_plus_news |         5 | 5979.8531 | 8465.7198 |               0.5124 |         0.1376 |          0.4778 |        -0.8494 |                        -0.0076 |

### Model: xgboost

| feature_set                    |   horizon |       mae |      rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:-------------------------------|----------:|----------:|----------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only                     |         1 | 2957.2687 | 5187.6599 |               0.5133 |         0.2772 |          0.3939 |        -0.4550 |                         0.0000 |
| price_plus_sentiment           |         1 | 4313.9143 | 6813.1103 |               0.5095 |         0.4388 |          0.6146 |        -0.4082 |                        -0.0038 |
| price_plus_sentiment_plus_news |         1 | 4237.9085 | 6716.3961 |               0.4943 |         0.1862 |          0.3174 |        -0.4673 |                        -0.0190 |
| price_only                     |         2 | 3100.1404 | 5369.1225 |               0.5228 |         0.8609 |          1.3956 |        -0.5710 |                         0.0000 |
| price_plus_sentiment           |         2 | 4317.9298 | 6776.4447 |               0.5105 |         0.4695 |          0.7910 |        -0.6122 |                        -0.0124 |
| price_plus_sentiment_plus_news |         2 | 4241.3776 | 6684.0821 |               0.5133 |         0.3502 |          0.5899 |        -0.6384 |                        -0.0095 |
| price_only                     |         3 | 3312.5069 | 5606.6022 |               0.5361 |         1.2843 |          2.1097 |        -0.6784 |                         0.0000 |
| price_plus_sentiment           |         3 | 4497.9701 | 6963.7336 |               0.5076 |         0.3197 |          0.5728 |        -0.6907 |                        -0.0285 |
| price_plus_sentiment_plus_news |         3 | 4468.5735 | 6892.8298 |               0.5086 |         0.4525 |          0.7524 |        -0.7275 |                        -0.0276 |
| price_only                     |         4 | 3582.0111 | 5879.1507 |               0.5143 |         0.6671 |          0.9967 |        -0.8510 |                         0.0000 |
| price_plus_sentiment           |         4 | 4846.9628 | 7276.3629 |               0.4990 |         0.0904 |          0.2352 |        -0.8024 |                        -0.0152 |
| price_plus_sentiment_plus_news |         4 | 4905.7669 | 7310.8674 |               0.5000 |         0.5023 |          0.9588 |        -0.8174 |                        -0.0143 |
| price_only                     |         5 | 3806.6858 | 6119.0116 |               0.5152 |         0.6128 |          0.9905 |        -0.8247 |                         0.0000 |
| price_plus_sentiment           |         5 | 5094.3656 | 7557.2331 |               0.5171 |         0.4110 |          0.7931 |        -0.8235 |                         0.0019 |
| price_plus_sentiment_plus_news |         5 | 5176.9163 | 7598.8123 |               0.5162 |         0.6238 |          1.2562 |        -0.8617 |                         0.0010 |

## Key research question

`delta_accuracy_vs_price_only` is the directional-accuracy improvement from adding X sentiment (and news) over price alone. If this is not consistently positive across horizons and folds, there is no evidence yet that sentiment adds forecasting value for this asset/period.