# BTC-USD — Model Evaluation Report

Generated: 2026-11-08T05:40:00.343712+00:00

## Methodology
- Walk-forward validation with an expanding training window; test folds always come strictly after training data.
- Three feature sets compared: price-only, price+X-sentiment, price+X-sentiment+news.
- Models: XGBoost, LightGBM, evaluated at horizons n+1..n+5.
- Trading metrics (Sharpe, Sortino, Max Drawdown) come from a simple long/short strategy: position direction = sign of predicted price change, realized return = actual price change over that horizon.

## Results

### Model: lightgbm

| feature_set                    |   horizon |      mae |     rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:-------------------------------|----------:|---------:|---------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only                     |         1 | 352.1833 | 515.9574 |               0.4948 |        -0.5425 |         -0.6032 |        -0.0930 |                         0.0000 |
| price_plus_sentiment           |         1 | 352.1833 | 515.9574 |               0.4948 |        -0.5425 |         -0.6032 |        -0.0930 |                         0.0000 |
| price_plus_sentiment_plus_news |         1 | 352.1833 | 515.9574 |               0.4948 |        -0.5425 |         -0.6032 |        -0.0930 |                         0.0000 |
| price_only                     |         2 | 417.4954 | 613.9764 |               0.4971 |        -1.2865 |         -1.2168 |        -0.1751 |                         0.0000 |
| price_plus_sentiment           |         2 | 417.4954 | 613.9764 |               0.4971 |        -1.2865 |         -1.2168 |        -0.1751 |                         0.0000 |
| price_plus_sentiment_plus_news |         2 | 417.4954 | 613.9764 |               0.4971 |        -1.2865 |         -1.2168 |        -0.1751 |                         0.0000 |
| price_only                     |         3 | 479.6451 | 696.1359 |               0.5133 |        -0.5430 |          0.5475 |        -0.2072 |                         0.0000 |
| price_plus_sentiment           |         3 | 479.6451 | 696.1359 |               0.5133 |        -0.5430 |          0.5475 |        -0.2072 |                         0.0000 |
| price_plus_sentiment_plus_news |         3 | 479.6451 | 696.1359 |               0.5133 |        -0.5430 |          0.5475 |        -0.2072 |                         0.0000 |
| price_only                     |         4 | 532.1657 | 768.4534 |               0.5070 |         0.6841 |          1.5636 |        -0.2622 |                         0.0000 |
| price_plus_sentiment           |         4 | 532.1657 | 768.4534 |               0.5070 |         0.6841 |          1.5636 |        -0.2622 |                         0.0000 |
| price_plus_sentiment_plus_news |         4 | 532.1657 | 768.4534 |               0.5070 |         0.6841 |          1.5636 |        -0.2622 |                         0.0000 |
| price_only                     |         5 | 583.7787 | 838.4852 |               0.4994 |        -2.3928 |         -1.9207 |        -0.3223 |                         0.0000 |
| price_plus_sentiment           |         5 | 583.7787 | 838.4852 |               0.4994 |        -2.3928 |         -1.9207 |        -0.3223 |                         0.0000 |
| price_plus_sentiment_plus_news |         5 | 583.7787 | 838.4852 |               0.4994 |        -2.3928 |         -1.9207 |        -0.3223 |                         0.0000 |

### Model: xgboost

| feature_set                    |   horizon |      mae |     rmse |   direction_accuracy |   sharpe_ratio |   sortino_ratio |   max_drawdown |   delta_accuracy_vs_price_only |
|:-------------------------------|----------:|---------:|---------:|---------------------:|---------------:|----------------:|---------------:|-------------------------------:|
| price_only                     |         1 | 418.6154 | 597.9365 |               0.4913 |        -2.8338 |         -3.0848 |        -0.0987 |                         0.0000 |
| price_plus_sentiment           |         1 | 421.9832 | 612.0802 |               0.4936 |        -1.8578 |         -2.2325 |        -0.0970 |                         0.0023 |
| price_plus_sentiment_plus_news |         1 | 400.5829 | 573.0863 |               0.4907 |        -1.2731 |         -1.4325 |        -0.1024 |                        -0.0006 |
| price_only                     |         2 | 451.1134 | 646.9776 |               0.4896 |        -2.5419 |         -2.6550 |        -0.1619 |                         0.0000 |
| price_plus_sentiment           |         2 | 455.1401 | 654.4483 |               0.4930 |        -2.5280 |         -2.9752 |        -0.1545 |                         0.0035 |
| price_plus_sentiment_plus_news |         2 | 439.6510 | 630.7390 |               0.4884 |        -0.9623 |         -0.5213 |        -0.1623 |                        -0.0012 |
| price_only                     |         3 | 519.4273 | 738.0604 |               0.5041 |        -3.3208 |         -3.6898 |        -0.2229 |                         0.0000 |
| price_plus_sentiment           |         3 | 520.3004 | 742.6986 |               0.5000 |        -2.6563 |         -2.8035 |        -0.2139 |                        -0.0041 |
| price_plus_sentiment_plus_news |         3 | 503.7446 | 717.8759 |               0.4936 |        -3.3466 |         -3.5471 |        -0.2289 |                        -0.0104 |
| price_only                     |         4 | 559.4905 | 791.2157 |               0.4896 |        -2.8821 |         -2.5819 |        -0.2592 |                         0.0000 |
| price_plus_sentiment           |         4 | 560.6118 | 797.3455 |               0.4884 |        -2.6190 |         -2.6897 |        -0.2665 |                        -0.0012 |
| price_plus_sentiment_plus_news |         4 | 548.2267 | 776.5049 |               0.4843 |        -2.6813 |         -2.4794 |        -0.2716 |                        -0.0052 |
| price_only                     |         5 | 602.2254 | 853.4326 |               0.4959 |        -2.5880 |         -2.5360 |        -0.3178 |                         0.0000 |
| price_plus_sentiment           |         5 | 603.2981 | 858.2732 |               0.4901 |        -3.7766 |         -4.1125 |        -0.3029 |                        -0.0058 |
| price_plus_sentiment_plus_news |         5 | 584.8296 | 829.3455 |               0.4977 |        -2.2121 |         -2.1323 |        -0.2888 |                         0.0017 |

## Key research question

`delta_accuracy_vs_price_only` is the directional-accuracy improvement from adding X sentiment (and news) over price alone. If this is not consistently positive across horizons and folds, there is no evidence yet that sentiment adds forecasting value for this asset/period.