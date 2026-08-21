import pandas as pd


def calculate_transaction_costs(
    positions: pd.Series,
    cost_bps_per_side: float = 5.0,
) -> pd.Series:
    """
    Calculate transaction costs based on changes
    in portfolio position.

    cost_bps_per_side:
        Transaction cost in basis points per unit
        of position turnover.

    Example:
        5 bps = 0.05%
    """

    turnover = positions.fillna(0).diff().abs()

    cost_rate = cost_bps_per_side / 10_000

    transaction_cost = turnover * cost_rate

    return transaction_cost
