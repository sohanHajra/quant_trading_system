import pandas as pd

from src.strategies.moving_average import generate_signals
from src.backtest.costs import calculate_transaction_costs


INITIAL_CAPITAL = 100_000

#initial backtest function

# def run_backtest(df: pd.DataFrame) -> pd.DataFrame:

#     data = generate_signals(df)

#     # IMPORTANT:
#     # Trade using tomorrow's position.
#     data["position"] = data["signal"].shift(1)

#     # Daily market return
#     data["market_return"] = data["close"].pct_change()

#     # Strategy return
#     data["strategy_return"] = (
#         data["position"] *
#         data["market_return"]
#     )

#     # Portfolio growth
#     data["equity"] = (
#         1 + data["strategy_return"]
#     ).cumprod()

#     data["portfolio_value"] = (
#         INITIAL_CAPITAL *
#         data["equity"]
#     )

#     return data



def run_backtest(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
    cost_bps_per_side: float = 5.0,
) -> pd.DataFrame:

    data = generate_signals(
    df,
    fast_window=fast_window,
    slow_window=slow_window,
)

    # Trade based on information available
    # from the previous observation.
    data["position"] = data["signal"].shift(1)

    # Daily market return
    data["market_return"] = data["close"].pct_change()

    # Gross strategy return before costs
    data["gross_return"] = (
        data["position"] *
        data["market_return"]
    )

    # Transaction costs
    data["transaction_cost"] = (
        calculate_transaction_costs(
            data["position"],
            cost_bps_per_side,
        )
    )

    # Net strategy return
    data["strategy_return"] = (
        data["gross_return"]
        - data["transaction_cost"]
    )

    # Portfolio growth
    data["equity"] = (
        1 + data["strategy_return"]
    ).cumprod()

    data["portfolio_value"] = (
        INITIAL_CAPITAL *
        data["equity"]
    )

    return data

