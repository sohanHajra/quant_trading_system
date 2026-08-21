import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
) -> pd.DataFrame:

    data = df.copy()

    data["sma_fast"] = (
        data["close"]
        .rolling(fast_window)
        .mean()
    )

    data["sma_slow"] = (
        data["close"]
        .rolling(slow_window)
        .mean()
    )

    data["signal"] = 0

    data.loc[
        data["sma_fast"] > data["sma_slow"],
        "signal"
    ] = 1

    return data
