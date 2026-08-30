import duckdb
import pandas as pd

from src.backtest.costs import calculate_transaction_costs
from src.backtest.simple_backtest import run_backtest
from src.analytics.performance import calculate_metrics


DATABASE_PATH = "data/market.duckdb"

INITIAL_CAPITAL = 100_000

TRAIN_MONTHS = 12
TEST_MONTHS = 3

FAST_WINDOWS = [10, 20, 30, 50]
SLOW_WINDOWS = [50, 100, 150, 200]

COST_BPS = 5.0

def load_market_data() -> pd.DataFrame:

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

def select_best_parameters(
    train_data: pd.DataFrame,
) -> dict:

    results = []

    for fast_window in FAST_WINDOWS:

        for slow_window in SLOW_WINDOWS:

            if fast_window >= slow_window:
                continue

            backtest = run_backtest(
                train_data,
                fast_window=fast_window,
                slow_window=slow_window,
                cost_bps_per_side=COST_BPS,
            )

            metrics = calculate_metrics(
                backtest
            )

            results.append(
                {
                    "fast_window": fast_window,
                    "slow_window": slow_window,
                    "sharpe": metrics[
                        "sharpe_ratio"
                    ],
                    "total_return": metrics[
                        "total_return"
                    ],
                }
            )

    results_df = pd.DataFrame(results)

    best = results_df.loc[
        results_df["sharpe"].idxmax()
    ]

    return {
        "fast_window": int(
            best["fast_window"]
        ),
        "slow_window": int(
            best["slow_window"]
        ),
    }
    
    
def generate_windows(
    df: pd.DataFrame,
):

    start_date = df["timestamp"].min()

    end_date = df["timestamp"].max()

    train_start = start_date

    while True:

        train_end = (
            train_start
            + pd.DateOffset(months=TRAIN_MONTHS)
        )

        test_end = (
            train_end
            + pd.DateOffset(months=TEST_MONTHS)
        )

        if test_end > end_date:
            break

        yield train_start, train_end, test_end

        train_start = (
            train_start
            + pd.DateOffset(months=TEST_MONTHS)
        )
        
def run_walk_forward(
    df: pd.DataFrame,
) -> pd.DataFrame:

    fold_results = []
    oos_returns = []

    for (
        train_start,
        train_end,
        test_end,
    ) in generate_windows(df):

        train_data = df[
            (df["timestamp"] >= train_start)
            & (df["timestamp"] < train_end)
        ].copy()

        # test_data = df[
        #     (df["timestamp"] >= train_end)
        #     & (df["timestamp"] < test_end)
        # ].copy()
        
        # accounting for warm up period
        slowest_window = max(SLOW_WINDOWS)

        test_start_index = df[
            df["timestamp"] >= train_end
        ].index[0]

        warmup_start_index = max(
            0,
            test_start_index - slowest_window
        )

        test_data = df.iloc[
            warmup_start_index:
        ][
            lambda x:
            (x["timestamp"] < test_end)
        ].copy()


        if train_data.empty or test_data.empty:
            continue

        best_params = select_best_parameters(
            train_data
        )

        
        # fixing the issue of indicator calculation
        
        # test_backtest = run_backtest(
        #     test_data,
        #     fast_window=best_params["fast_window"],
        #     slow_window=best_params["slow_window"],
        #     cost_bps_per_side=COST_BPS,
        # )
        
        # test_backtest = test_backtest[
        #     test_backtest["timestamp"] >= train_end
        # ].copy()


        # test_metrics = calculate_metrics(
        #     test_backtest
        # )
        
        
        # changes made to fix the issue of indicator calculation in the out-of-sample period
        
        # test_backtest = run_backtest(
        #     test_data,
        #     fast_window=best_params["fast_window"],
        #     slow_window=best_params["slow_window"],
        #     cost_bps_per_side=COST_BPS,
        # )

        # # Keep only the true out-of-sample period.
        # test_backtest = test_backtest[
        #     test_backtest["timestamp"] >= train_end
        # ].copy()

        # # Reset the portfolio to $100,000 at the
        # # beginning of the out-of-sample period.
        # test_backtest["equity"] = (
        #     1 + test_backtest["strategy_return"]
        # ).cumprod()

        # test_backtest["portfolio_value"] = (
        #     INITIAL_CAPITAL *
        #     test_backtest["equity"]
        # )

        # test_metrics = calculate_metrics(
        #     test_backtest,
        #     initial_capital=INITIAL_CAPITAL,
        # )


        # edit to the previous approach to reset the oos portfolio value to $100,000 at the beginning of the out-of-sample period while keeping the indicator calculations intact
        
        
        test_backtest = run_backtest(
            test_data,
            fast_window=best_params["fast_window"],
            slow_window=best_params["slow_window"],
            cost_bps_per_side=COST_BPS,
        )
        
        fast_window = best_params["fast_window"]
        slow_window = best_params["slow_window"]

        test_backtest["fast_ma"] = (
            test_backtest["close"]
            .rolling(
                window=fast_window,
                min_periods=fast_window,
            )
            .mean()
        )

        test_backtest["slow_ma"] = (
            test_backtest["close"]
            .rolling(
                window=slow_window,
                min_periods=slow_window,
            )
            .mean()
        )

        # Keep only the true out-of-sample period
        test_backtest = test_backtest[
            test_backtest["timestamp"] >= train_end
        ].copy()

        # The OOS portfolio begins flat, so the first position should be 0
        test_backtest.iloc[0, test_backtest.columns.get_loc("position")] = 0

        # Recalculate the return chain from the corrected position
        test_backtest["gross_return"] = (
            test_backtest["position"]
            * test_backtest["market_return"]
        )

        test_backtest["transaction_cost"] = (
            calculate_transaction_costs(
                test_backtest["position"],
                COST_BPS,
            )
        )
        
        # The first OOS observation starts with a flat portfolio
        # There is no prior OOS position, so there is no transaction cost
        test_backtest.iloc[
            0,
            test_backtest.columns.get_loc("transaction_cost")
        ] = 0.0

        test_backtest["strategy_return"] = (
            test_backtest["gross_return"]
            - test_backtest["transaction_cost"]
        )

        # Start the OOS equity curve from $100,000
        test_backtest["equity"] = (
            1 + test_backtest["strategy_return"]
        ).cumprod()

        test_backtest["portfolio_value"] = (
            INITIAL_CAPITAL *
            test_backtest["equity"]
        )

        test_metrics = calculate_metrics(
            test_backtest,
            initial_capital=INITIAL_CAPITAL,
        )
        
        #added position column to the oos_returns DataFrame to keep track of the position during the out-of-sample period
        #added close column to the oos_returns DataFrame to keep track of the close price during the out-of-sample period
        
        # oos_returns.append(
        #     test_backtest[
        #         [
        #             "timestamp",
        #             "close",
        #             "strategy_return",
        #             "position",
        #             "portfolio_value",
        #         ]
        #     ].copy()
        # )
        
        #----------------------------------------------------
        oos_fold = test_backtest[
            [
                "timestamp",
                "close",
                "market_return",
                "strategy_return",
                "position",
                "fast_ma",
                "slow_ma",
            ]
        ].copy()

        # preserve the parameters actually selected for this fold
        oos_fold["fast_window"] = best_params["fast_window"]
        oos_fold["slow_window"] = best_params["slow_window"]

        oos_returns.append(oos_fold)
        

        fold_results.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_end": test_end,
                "fast_window": best_params[
                    "fast_window"
                ],
                "slow_window": best_params[
                    "slow_window"
                ],
                "test_return": test_metrics[
                    "total_return"
                ],
                "test_sharpe": test_metrics[
                    "sharpe_ratio"
                ],
                "test_drawdown": test_metrics[
                    "max_drawdown"
                ],
                "test_trades": test_metrics[
                    "trade_count"
                ],
            }
        )

    # return (pd.DataFrame(fold_results), pd.concat(oos_returns, ignore_index=True))
    
    # more robust way to handle the case when there are no complete walk-forward windows
    # added position column to the combined_oos DataFrame to keep track of the position during the out-of-sample period
    #added close column to the combined_oos DataFrame to keep track of the close price during the out-of-sample period
    if not oos_returns:
        combined_oos = pd.DataFrame(
            columns=[
                "timestamp",
                "close",
                "market_return",
                "strategy_return",
                "position",
                "fast_ma",
                "slow_ma",
                "fast_window",
                "slow_window",
            ]
        )
    else:
        combined_oos = pd.concat(
            oos_returns,
            ignore_index=True,
        )

    return (
        pd.DataFrame(fold_results),
        combined_oos,
    )


def main():

    df = load_market_data()

    results, oos_returns = run_walk_forward(df)
    
    # added to get the contribution of each fold to the overall out-of-sample performance
    oos_data = (
        oos_data
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    oos_data["equity"] = (
        1 + oos_data["strategy_return"]
    ).cumprod()

    oos_data["portfolio_value"] = (
        100_000
        * oos_data["equity"]
    )
    
    #combined portfolio value calculation for the out-of-sample period
    oos_returns = (
        oos_returns
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    oos_returns["equity"] = (
        1 + oos_returns["strategy_return"]
    ).cumprod()

    oos_returns["portfolio_value"] = (
        INITIAL_CAPITAL
        * oos_returns["equity"]
    )
    
    #oos performance calculation with the sharpe bug and also without CAGR
    # combined_oos_return = (
    #     oos_returns["portfolio_value"].iloc[-1]
    #     / INITIAL_CAPITAL
    # ) - 1

    # combined_oos_volatility = (
    #     oos_returns["strategy_return"].std()
    #     * (252 ** 0.5)
    # )

    # combined_oos_sharpe = (
    #     combined_oos_return
    #     / combined_oos_volatility
    #     if combined_oos_volatility != 0
    #     else float("nan")
    # )
    
    
    # ============================================================
    # Combined OOS performance metrics
    # ============================================================

    final_value = oos_returns["portfolio_value"].iloc[-1]

    # total cumulative return over the entire OOS period
    combined_oos_return = (
        final_value / INITIAL_CAPITAL
    ) - 1


    # ------------------------------------------------------------
    # Annualized return (CAGR)
    # ------------------------------------------------------------

    start_date = oos_returns["timestamp"].iloc[0]
    end_date = oos_returns["timestamp"].iloc[-1]

    days = (
        end_date - start_date
    ).days

    years = days / 365.25

    combined_oos_cagr = (
        (final_value / INITIAL_CAPITAL)
        ** (1 / years)
    ) - 1


    # ------------------------------------------------------------
    # Annualized volatility
    # ------------------------------------------------------------

    daily_returns = (
        oos_returns["strategy_return"]
        .dropna()
    )

    daily_volatility = daily_returns.std()

    combined_oos_volatility = (
        daily_volatility * (252 ** 0.5)
    )


    # ------------------------------------------------------------
    # Sharpe ratio (Assumes a zero risk-free rate for now)
    # ------------------------------------------------------------

    daily_mean_return = daily_returns.mean()

    combined_oos_sharpe = (
        (daily_mean_return / daily_volatility)
        * (252 ** 0.5)
        if daily_volatility != 0
        else float("nan")
    )
    
    
    # ============================================================
    # Running maximum
    # ============================================================
    running_max = (
        oos_returns["portfolio_value"]
        .cummax()
    )

    drawdown = (
        oos_returns["portfolio_value"]
        / running_max
    ) - 1

    combined_oos_drawdown = drawdown.min()
    
    # calculating the exposure of the strategy during the out-of-sample period
    exposure = (oos_returns["position"] != 0).mean()
    
    
    #Results summary
    print(
        "\n========== WALK-FORWARD RESULTS =========="
    )

    if results.empty:
        print(
            "No complete walk-forward windows found."
        )
        return

    print(
        results.to_string(index=False)
    )

    print(
        "\n========== OUT-OF-SAMPLE SUMMARY =========="
    )

    print(
        f"Number of folds: "
        f"{len(results)}"
    )

    print(
        f"Average test return: "
        f"{results['test_return'].mean():.2%}"
    )

    print(
        f"Average test Sharpe: "
        f"{results['test_sharpe'].mean():.2f}"
    )

    print(
        f"Average test drawdown: "
        f"{results['test_drawdown'].mean():.2%}"
    )
    
    
    print(
        "\n========== COMBINED OOS PERFORMANCE =========="
    )

    print(
    f"Final OOS portfolio value: "
    f"${final_value:,.2f}"
    )

    print(
        f"Combined OOS return: "
        f"{combined_oos_return:.2%}"
    )

    print(
        f"Combined OOS CAGR: "
        f"{combined_oos_cagr:.2%}"
    )

    print(
        f"Combined OOS volatility: "
        f"{combined_oos_volatility:.2%}"
    )

    print(
        f"Combined OOS Sharpe: "
        f"{combined_oos_sharpe:.2f}"
    )

    print(
        f"Combined OOS max drawdown: "
        f"{combined_oos_drawdown:.2%}"
    )
    
    print(
        f"Strategy exposure: "
        f"{exposure:.2%}"
    )

if __name__ == "__main__":
    main()
