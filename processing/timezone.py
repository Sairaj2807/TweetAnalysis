import pandas as pd

DISPLAY_TIMEZONE = "UTC"  # swap to "America/New_York" or "Asia/Kolkata" if needed for reporting


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def floor_to_interval(series: pd.Series, interval: str) -> pd.Series:
    """Floors timestamps to the candle interval boundary, e.g. '1h'."""
    return pd.to_datetime(series, utc=True).dt.floor(interval)


def to_display_timezone(series: pd.Series, tz: str = DISPLAY_TIMEZONE) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(tz)
