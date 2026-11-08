from lightgbm import LGBMRegressor


def build_lightgbm_model(**kwargs) -> LGBMRegressor:
    params = dict(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    params.update(kwargs)
    return LGBMRegressor(**params)
