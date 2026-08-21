import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calculate_metrics(
    results: pd.DataFrame,
    initial_capital: float = 100_000,
) -> dict:

    returns = results["strategy_return"].dropna()

    final_value = results["portfolio_value"].iloc[-1]

    total_return = (
        final_value / initial_capital
    ) - 1

    # Number of years represented by the data
    days = (
        results["timestamp"].iloc[-1]
        - results["timestamp"].iloc[0]
    ).days

    years = days / 365.25

    annualized_return = (
        (final_value / initial_capital)
        ** (1 / years)
    ) - 1

    annualized_volatility = (
        returns.std() * np.sqrt(TRADING_DAYS)
    )

    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility != 0
        else np.nan
    )

    # Drawdown
    equity = results["portfolio_value"]

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
    ) - 1

    max_drawdown = drawdown.min()

    # Number of position changes
    trades = (
        results["position"]
        .fillna(0)
        .diff()
        .abs()
        .sum()
    )

    return {
        "final_value": final_value,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "trade_count": int(trades),
    }
