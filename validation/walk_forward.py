import pandas as pd

from config.settings import TEST_FRAC, TRAIN_FRAC, VAL_FRAC


def chronological_split(df: pd.DataFrame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Simple single chronological split: train / val / test, in time order (no shuffling)."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return train, val, test


def walk_forward_windows(df: pd.DataFrame, n_splits: int = 4, min_train_frac: float = 0.5, test_frac: float = 0.1, purge: int = 0):
    """Yields (train_df, test_df) pairs with an expanding training window rolled forward in time.

    Each fold's test window comes strictly after its training window -- no future
    information ever leaks into training, and folds are evaluated out-of-sample.

    `purge` drops the last `purge` rows before the test window from training. This
    is needed because targets are built via price.shift(-h): without a purge, the
    last `h` training rows have targets that fall inside the test window, so the
    model trains on information from the "future" test period. Pass
    purge=max(HORIZONS) so no horizon's target can reach into any test window.
    assert_no_overlap() only checks timestamps, not target reach, so it does not
    catch this on its own -- purge is the actual fix.
    """
    n = len(df)
    min_train_size = int(n * min_train_frac)
    test_size = int(n * test_frac)
    remaining = n - min_train_size
    step = max(test_size, remaining // n_splits) if n_splits > 0 else test_size

    start = min_train_size
    while start + test_size <= n:
        train_df = df.iloc[: max(0, start - purge)].reset_index(drop=True)
        test_df = df.iloc[start : start + test_size].reset_index(drop=True)
        yield train_df, test_df
        start += step
