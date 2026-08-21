import matplotlib.pyplot as plt
import pandas as pd

from src.research.walk_forward import (
    load_market_data,
    run_walk_forward,
)


INITIAL_CAPITAL = 100_000


def main():

    # --------------------------------------------------------
    # Load market data
    # --------------------------------------------------------

    df = load_market_data()

    # --------------------------------------------------------
    # Run the walk-forward research
    # --------------------------------------------------------

    results, oos_returns = run_walk_forward(df)

    if oos_returns.empty:
        print("No out-of-sample data found.")
        return

    # --------------------------------------------------------
    # Sort OOS observations chronologically
    # --------------------------------------------------------

    oos_returns = (
        oos_returns
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Build strategy equity curve
    # --------------------------------------------------------

    oos_returns["strategy_equity"] = (
        1 + oos_returns["strategy_return"]
    ).cumprod()

    oos_returns["strategy_value"] = (
        INITIAL_CAPITAL
        * oos_returns["strategy_equity"]
    )

    # --------------------------------------------------------
    # Get SPY prices for the same OOS dates
    # --------------------------------------------------------

    benchmark = df[
        ["timestamp", "close"]
    ].copy()

    benchmark = benchmark[
        benchmark["timestamp"].isin(
            oos_returns["timestamp"]
        )
    ].copy()

    benchmark = (
        benchmark
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Normalize SPY to the same starting capital
    # --------------------------------------------------------

    first_close = benchmark["close"].iloc[0]

    benchmark["spy_value"] = (
        INITIAL_CAPITAL
        * benchmark["close"]
        / first_close
    )

    # --------------------------------------------------------
    # Merge the two equity curves
    # --------------------------------------------------------

    comparison = pd.merge(
        oos_returns[
            [
                "timestamp",
                "strategy_value",
            ]
        ],
        benchmark[
            [
                "timestamp",
                "spy_value",
            ]
        ],
        on="timestamp",
        how="inner",
    )

    # --------------------------------------------------------
    # Plot equity curves
    # --------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        comparison["timestamp"],
        comparison["strategy_value"],
        label="Walk-Forward Strategy",
    )

    plt.plot(
        comparison["timestamp"],
        comparison["spy_value"],
        label="SPY Buy & Hold",
    )

    plt.title(
        "Out-of-Sample Strategy vs SPY"
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------
    # Calculate drawdowns
    # --------------------------------------------------------

    strategy_running_max = (
        comparison["strategy_value"]
        .cummax()
    )

    strategy_drawdown = (
        comparison["strategy_value"]
        / strategy_running_max
    ) - 1

    spy_running_max = (
        comparison["spy_value"]
        .cummax()
    )

    spy_drawdown = (
        comparison["spy_value"]
        / spy_running_max
    ) - 1

    # --------------------------------------------------------
    # Plot drawdowns
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        comparison["timestamp"],
        strategy_drawdown,
        label="Walk-Forward Strategy",
    )

    plt.plot(
        comparison["timestamp"],
        spy_drawdown,
        label="SPY Buy & Hold",
    )

    plt.title(
        "Out-of-Sample Drawdown Comparison"
    )

    plt.xlabel("Date")
    plt.ylabel("Drawdown")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()