import pandas as pd


REQUIRED_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate basic OHLCV market data."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("DataFrame is empty.")

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Data is not sorted chronologically."
        )

    if df.index.duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected."
        )

    if (df["high"] < df["low"]).any():
        raise ValueError(
            "Found high price below low price."
        )

    if (df["volume"] < 0).any():
        raise ValueError(
            "Negative volume detected."
        )

    print("Data validation passed.")
