import pandas as pd


def assert_no_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, time_col: str = "timestamp_utc") -> None:
    """Fails loudly if any test timestamp is <= the latest training timestamp."""
    if train_df.empty or test_df.empty:
        return
    max_train_time = pd.to_datetime(train_df[time_col]).max()
    min_test_time = pd.to_datetime(test_df[time_col]).min()
    if min_test_time <= max_train_time:
        raise ValueError(
            f"Leakage detected: test window starts at {min_test_time} but training data "
            f"extends to {max_train_time}."
        )


def assert_scaler_fit_on_train_only(scaler_fit_data_source: str) -> None:
    """Documents the invariant; call-sites should pass 'train' to make the intent explicit."""
    if scaler_fit_data_source != "train":
        raise ValueError("Scalers/preprocessors must be fit on training data only.")
