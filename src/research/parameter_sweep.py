import duckdb
import pandas as pd

from src.backtest.simple_backtest import run_backtest
from src.analytics.performance import calculate_metrics


DATABASE_PATH = "data/market.duckdb"


def load_market_data() -> pd.DataFrame:
    """Load historical prices from DuckDB."""

    con = duckdb.connect(DATABASE_PATH)

    df = con.execute(
        """
        SELECT *
        FROM prices
        ORDER BY timestamp
        """
    ).fetchdf()

    con.close()

    return df


def run_parameter_sweep(
    df: pd.DataFrame,
    fast_windows: list[int],
    slow_windows: list[int],
    cost_bps_per_side: float = 5.0,
) -> pd.DataFrame:

    results = []

    for fast_window in fast_windows:

        for slow_window in slow_windows:

            # A fast moving average should be
            # shorter than the slow moving average.
            if fast_window >= slow_window:
                continue

            backtest = run_backtest(
                df,
                fast_window=fast_window,
                slow_window=slow_window,
                cost_bps_per_side=cost_bps_per_side,
            )

            metrics = calculate_metrics(
                backtest
            )

            results.append(
                {
                    "fast_window": fast_window,
                    "slow_window": slow_window,
                    "total_return": metrics[
                        "total_return"
                    ],
                    "annualized_return": metrics[
                        "annualized_return"
                    ],
                    "volatility": metrics[
                        "annualized_volatility"
                    ],
                    "sharpe": metrics[
                        "sharpe_ratio"
                    ],
                    "max_drawdown": metrics[
                        "max_drawdown"
                    ],
                    "trade_count": metrics[
                        "trade_count"
                    ],
                }
            )

    return pd.DataFrame(results)


def main():

    df = load_market_data()

    fast_windows = [10, 20, 30, 50]

    slow_windows = [50, 100, 150, 200]

    results = run_parameter_sweep(
        df,
        fast_windows,
        slow_windows,
        cost_bps_per_side=5.0,
    )

    results = results.sort_values(
        "sharpe",
        ascending=False,
    )

    print(
        "\n========== PARAMETER SWEEP =========="
    )

    print(
        results.to_string(index=False)
    )


if __name__ == "__main__":
    main()
