from xgboost import XGBRegressor


def build_xgboost_model(**kwargs) -> XGBRegressor:
    params = dict(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    params.update(kwargs)
    return XGBRegressor(**params)
