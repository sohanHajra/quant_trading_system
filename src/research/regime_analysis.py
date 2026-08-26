import pandas as pd

from src.research.walk_forward import (
    load_market_data,
    run_walk_forward,
)


def classify_regime(
    trend_strength: float,
) -> str:

    if trend_strength >= 0.02:
        return "strong_uptrend"

    if trend_strength >= 0.005:
        return "weak_uptrend"

    if trend_strength > -0.005:
        return "neutral"

    if trend_strength > -0.02:
        return "weak_downtrend"

    return "strong_downtrend"



def add_regime_features(
    oos_data: pd.DataFrame,
) -> pd.DataFrame:

    data = (
        oos_data
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    data["trend_strength"] = (
        data["fast_ma"]
        - data["slow_ma"]
    ) / data["slow_ma"]

    data["regime"] = (
        data["trend_strength"]
        .apply(classify_regime)
    )

    return data


def main():

    df = load_market_data()

    results, oos_data = run_walk_forward(df)

    if oos_data.empty:
        print("No OOS data available.")
        return

    oos_data = (
        oos_data
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    # making it cleaner and simpler to add regime features to the OOS data
    oos_data = add_regime_features(
        oos_data
    )
    
    
    print("\n========== REGIME SUMMARY ==========")

    print(
        oos_data["regime"]
        .value_counts()
        .sort_index()
    )
    
    
if __name__ == "__main__":
    main()