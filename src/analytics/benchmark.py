import numpy as np
import pandas as pd

from src.research.walk_forward import (
    load_market_data,
    run_walk_forward,
)


INITIAL_CAPITAL = 100_000
TRADING_DAYS = 252


def calculate_performance(
    returns: pd.Series,
    portfolio_value: pd.Series,
    initial_capital: float,
) -> dict:

    returns = returns.dropna()

    final_value = portfolio_value.iloc[-1]

    total_return = (
        final_value / initial_capital
    ) - 1

    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    # We need the actual date range separately.
    years = (
        portfolio_value.index[-1]
        - portfolio_value.index[0]
    ).days / 365.25

    cagr = (
        (final_value / initial_capital)
        ** (1 / years)
    ) - 1

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    daily_volatility = returns.std()

    annualized_volatility = (
        daily_volatility
        * np.sqrt(TRADING_DAYS)
    )

    # --------------------------------------------------------
    # Sharpe ratio
    # --------------------------------------------------------

    daily_mean_return = returns.mean()

    sharpe = (
        (
            daily_mean_return
            / daily_volatility
        )
        * np.sqrt(TRADING_DAYS)
        if daily_volatility != 0
        else np.nan
    )

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    running_max = portfolio_value.cummax()

    drawdown = (
        portfolio_value
        / running_max
    ) - 1

    max_drawdown = drawdown.min()

    return {
        "final_value": final_value,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def main():

    # ========================================================
    # Load data and generate OOS strategy results
    # ========================================================

    df = load_market_data()

    results, oos_returns = run_walk_forward(df)

    if oos_returns.empty:
        print("No out-of-sample data found.")
        return

    # ========================================================
    # Sort OOS observations chronologically
    # ========================================================

    oos_returns = (
        oos_returns
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ========================================================
    # Strategy portfolio value
    # ========================================================

    oos_returns["strategy_equity"] = (
        1 + oos_returns["strategy_return"]
    ).cumprod()

    oos_returns["strategy_value"] = (
        INITIAL_CAPITAL
        * oos_returns["strategy_equity"]
    )

    # ========================================================
    # Calculate strategy metrics
    # ========================================================

    strategy_returns = (
        oos_returns["strategy_return"]
    )

    strategy_portfolio = (
        oos_returns
        .set_index("timestamp")["strategy_value"]
    )

    strategy_metrics = calculate_performance(
        strategy_returns,
        strategy_portfolio,
        INITIAL_CAPITAL,
    )

    # ========================================================
    # Build SPY buy-and-hold benchmark
    # ========================================================

    benchmark = df[
        [
            "timestamp",
            "close",
        ]
    ].copy()

    # Restrict SPY to the exact OOS period
    benchmark = benchmark[
        (benchmark["timestamp"] >= oos_returns["timestamp"].min())
        & (benchmark["timestamp"] <= oos_returns["timestamp"].max())
    ].copy()

    benchmark = (
        benchmark
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Daily SPY returns
    benchmark["spy_return"] = (
        benchmark["close"].pct_change()
    )

    # Normalize SPY to $100,000
    benchmark["spy_equity"] = (
        1 + benchmark["spy_return"].fillna(0)
    ).cumprod()

    benchmark["spy_value"] = (
        INITIAL_CAPITAL
        * benchmark["spy_equity"]
    )

    # ========================================================
    # Calculate SPY metrics
    # ========================================================

    spy_returns = benchmark["spy_return"]

    spy_portfolio = (
        benchmark
        .set_index("timestamp")["spy_value"]
    )

    spy_metrics = calculate_performance(
        spy_returns,
        spy_portfolio,
        INITIAL_CAPITAL,
    )

    # ========================================================
    # Print comparison
    # ========================================================

    print("\n========== BENCHMARK COMPARISON ==========")

    comparison = pd.DataFrame(
        {
            "Strategy": [
                strategy_metrics["final_value"],
                strategy_metrics["total_return"],
                strategy_metrics["cagr"],
                strategy_metrics["volatility"],
                strategy_metrics["sharpe"],
                strategy_metrics["max_drawdown"],
            ],
            "SPY Buy & Hold": [
                spy_metrics["final_value"],
                spy_metrics["total_return"],
                spy_metrics["cagr"],
                spy_metrics["volatility"],
                spy_metrics["sharpe"],
                spy_metrics["max_drawdown"],
            ],
        },
        index=[
            "Final Value",
            "Total Return",
            "CAGR",
            "Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
        ],
    )

    print(comparison)
    

if __name__ == "__main__":
    main()