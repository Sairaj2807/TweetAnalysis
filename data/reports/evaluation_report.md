# BTC-USD — Model Evaluation Report

Generated: 2026-11-08T04:05:35.105365+00:00

## Methodology
- Walk-forward validation with an expanding training window; test folds always come strictly after training data.
- Three feature sets compared: price-only, price+X-sentiment, price+X-sentiment+news.
- Models: XGBoost, LightGBM, evaluated at horizons n+1..n+5.
- Trading metrics (Sharpe, Sortino, Max Drawdown) come from a simple long/short strategy: position direction = sign of predicted price change, realized return = actual price change over that horizon.

## Results

### Model: lightgbm

| feature_set                    |   horizon |      mae |     rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:-------------------------------|----------:|---------:|---------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only                     |         1 | 346.4112 | 502.6505 |               0.5023 |         0.3384 |          0.6328 |        -0.0863 |                         0.0000 |
| price_plus_sentiment           |         1 | 346.4112 | 502.6505 |               0.5023 |         0.3384 |          0.6328 |        -0.0863 |                         0.0000 |
| price_plus_sentiment_plus_news |         1 | 346.4112 | 502.6505 |               0.5023 |         0.3384 |          0.6328 |        -0.0863 |                         0.0000 |
| price_only                     |         2 | 412.9613 | 600.1662 |               0.5041 |        -0.1409 |         -0.0133 |        -0.1469 |                         0.0000 |
| price_plus_sentiment           |         2 | 412.9613 | 600.1662 |               0.5041 |        -0.1409 |         -0.0133 |        -0.1469 |                         0.0000 |
| price_plus_sentiment_plus_news |         2 | 412.9613 | 600.1662 |               0.5041 |        -0.1409 |         -0.0133 |        -0.1469 |                         0.0000 |
| price_only                     |         3 | 477.4915 | 689.0256 |               0.5052 |        -2.5329 |         -2.4742 |        -0.2438 |                         0.0000 |
| price_plus_sentiment           |         3 | 477.4915 | 689.0256 |               0.5052 |        -2.5329 |         -2.4742 |        -0.2438 |                         0.0000 |
| price_plus_sentiment_plus_news |         3 | 477.4915 | 689.0256 |               0.5052 |        -2.5329 |         -2.4742 |        -0.2438 |                         0.0000 |
| price_only                     |         4 | 535.1332 | 771.2480 |               0.5058 |         0.2237 |          0.9550 |        -0.2614 |                         0.0000 |
| price_plus_sentiment           |         4 | 535.1332 | 771.2480 |               0.5058 |         0.2237 |          0.9550 |        -0.2614 |                         0.0000 |
| price_plus_sentiment_plus_news |         4 | 535.1332 | 771.2480 |               0.5058 |         0.2237 |          0.9550 |        -0.2614 |                         0.0000 |
| price_only                     |         5 | 583.2747 | 836.3904 |               0.5087 |        -0.0265 |          0.3567 |        -0.2839 |                         0.0000 |
| price_plus_sentiment           |         5 | 583.2747 | 836.3904 |               0.5087 |        -0.0265 |          0.3567 |        -0.2839 |                         0.0000 |
| price_plus_sentiment_plus_news |         5 | 583.2747 | 836.3904 |               0.5087 |        -0.0265 |          0.3567 |        -0.2839 |                         0.0000 |

### Model: xgboost

| feature_set                    |   horizon |      mae |     rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:-------------------------------|----------:|---------:|---------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only                     |         1 | 427.9778 | 610.1221 |               0.4919 |        -1.3670 |         -1.3105 |        -0.0967 |                         0.0000 |
| price_plus_sentiment           |         1 | 424.9785 | 613.5770 |               0.4971 |        -0.0965 |         -0.1574 |        -0.1073 |                         0.0052 |
| price_plus_sentiment_plus_news |         1 | 415.0534 | 591.5058 |               0.4913 |        -1.5893 |         -1.9753 |        -0.0935 |                        -0.0006 |
| price_only                     |         2 | 471.9455 | 671.9863 |               0.4913 |        -1.8414 |         -2.0476 |        -0.1605 |                         0.0000 |
| price_plus_sentiment           |         2 | 475.1410 | 678.0336 |               0.4878 |        -0.6916 |         -0.9857 |        -0.1558 |                        -0.0035 |
| price_plus_sentiment_plus_news |         2 | 458.9012 | 655.6978 |               0.4826 |        -2.4228 |         -3.0110 |        -0.1685 |                        -0.0087 |
| price_only                     |         3 | 523.5856 | 742.6560 |               0.5035 |        -1.4705 |         -0.8694 |        -0.2298 |                         0.0000 |
| price_plus_sentiment           |         3 | 538.3781 | 761.7780 |               0.5052 |        -1.8666 |         -1.6828 |        -0.2272 |                         0.0017 |
| price_plus_sentiment_plus_news |         3 | 517.8213 | 734.6096 |               0.4988 |        -1.0645 |         -0.3938 |        -0.2237 |                        -0.0046 |
| price_only                     |         4 | 584.7982 | 823.4365 |               0.4884 |        -1.3655 |         -1.1107 |        -0.2535 |                         0.0000 |
| price_plus_sentiment           |         4 | 582.2264 | 814.7472 |               0.4809 |        -1.8811 |         -1.9410 |        -0.2732 |                        -0.0075 |
| price_plus_sentiment_plus_news |         4 | 568.8509 | 799.6898 |               0.4820 |        -1.7758 |         -1.5859 |        -0.2697 |                        -0.0064 |
| price_only                     |         5 | 622.5744 | 876.8737 |               0.4919 |        -2.5075 |         -2.1694 |        -0.2965 |                         0.0000 |
| price_plus_sentiment           |         5 | 637.3461 | 890.4790 |               0.4890 |        -2.1012 |         -2.1286 |        -0.3055 |                        -0.0029 |
| price_plus_sentiment_plus_news |         5 | 611.9498 | 863.9802 |               0.4942 |        -2.0065 |         -1.7280 |        -0.3002 |                         0.0023 |

## Key research question

`delta_accuracy_vs_price_only` is the directional-accuracy improvement from adding X sentiment (and news) over price alone. If this is not consistently positive across horizons and folds, there is no evidence yet that sentiment adds forecasting value for this asset/period.