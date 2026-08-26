import pandas as pd
from src.research.walk_forward import (
    load_market_data,
    run_walk_forward,
)
from src.research.regime_analysis import (
    add_regime_features,
)


def extract_trades(
    oos_data: pd.DataFrame,
) -> pd.DataFrame:

    data = (
        oos_data
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    trades = []

    # added entry index to keep track of the index of the entry point in the data DataFrame    
    in_trade = False
    entry_index = None
    entry_date = None
    entry_price = None

    for i in range(len(data)):

        position = data.loc[i, "position"]
        price = data.loc[i, "close"]

        # ----------------------------------------------------
        # Entry
        # ----------------------------------------------------

        if not in_trade and position == 1:

            in_trade = True
            entry_index = i
            entry_date = data.loc[i, "timestamp"]
            entry_price = price
            entry_regime = data.loc[
                i,
                "regime",
            ]

            entry_trend_strength = data.loc[
                i,
                "trend_strength",
            ]
                        
            entry_portfolio_value = (
                data.loc[i - 1, "portfolio_value"]
                if i > 0
                else 100_000
            )

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        elif in_trade and position == 0:

            exit_index = i
            
            trade_trend_strength = (
                data.loc[
                    entry_index:exit_index,
                    "trend_strength",
                ]
            )

            average_trend_strength = (
                trade_trend_strength.mean()
            )
            
            average_regime = (
                data.loc[
                    entry_index:exit_index,
                    "regime",
                ]
                .mode()
                .iloc[0]
            )
            
            exit_date = data.loc[i, "timestamp"]
            exit_price = price

            trade_return = (
                exit_price / entry_price
            ) - 1

            portfolio_return = (
                1 + data.loc[
                    entry_index:exit_index,
                    "strategy_return"
                ]
            ).prod() - 1

            if entry_index == 0:
                starting_portfolio = 100_000
            else:
                starting_portfolio = (
                    data.loc[
                        entry_index - 1,
                        "portfolio_value"
                    ]
                )

            portfolio_pnl = (
                starting_portfolio
                * portfolio_return
            )

            holding_days = (
                exit_date - entry_date
            ).days
            

            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "trade_return": trade_return,
                    "portfolio_return": portfolio_return,
                    "portfolio_pnl": portfolio_pnl,
                    "holding_days": holding_days,
                    "entry_regime": entry_regime,
                    "entry_trend_strength": entry_trend_strength,
                    "average_regime": average_regime,
                    "average_trend_strength": average_trend_strength,
                    "exit_reason": "signal",
                }
            )

            in_trade = False
            entry_index = None
            entry_date = None
            entry_price = None

    # --------------------------------------------------------
    # Handle an open position at the end of the OOS sample
    # --------------------------------------------------------

    if in_trade:

        exit_index = len(data) - 1

        exit_date = data.iloc[-1]["timestamp"]
        exit_price = data.iloc[-1]["close"]

        trade_return = (
            exit_price / entry_price
        ) - 1

        portfolio_return = (
            1 + data.loc[
                entry_index:exit_index,
                "strategy_return"
            ]
        ).prod() - 1

        if entry_index == 0:
            starting_portfolio = 100_000
        else:
            starting_portfolio = (
                data.loc[
                    entry_index - 1,
                    "portfolio_value"
                ]
            )

        portfolio_pnl = (
            starting_portfolio
            * portfolio_return
        )

        holding_days = (
            exit_date - entry_date
        ).days
        

        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "trade_return": trade_return,
                "portfolio_return": portfolio_return,
                "portfolio_pnl": portfolio_pnl,
                "holding_days": holding_days,
                "entry_regime": entry_regime,
                "entry_trend_strength": entry_trend_strength,
                "average_regime": average_regime,
                "average_trend_strength": average_trend_strength,
                "exit_reason": "end_of_sample",
            }
        )

    return pd.DataFrame(trades)



def main():

    df = load_market_data()

    results, oos_data = run_walk_forward(df)

    if oos_data.empty:
        print("No OOS data available.")
        return
    
    # sort the OOS data chronologically
    oos_data = (
        oos_data
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    # reconstruct one continuous OOS portfolio
    oos_data["equity"] = (
        1 + oos_data["strategy_return"]
    ).cumprod()

    oos_data["portfolio_value"] = (
        100_000
        * oos_data["equity"]
    )
    
    # add regime features to the OOS data
    oos_data = add_regime_features(
        oos_data
    )
    
    # added this check to ensure that the portfolio value is not NaN at any point in the OOS data
    # #--------------------------------------------------------
    # print("\n========== PORTFOLIO VALUE DIAGNOSTIC ==========")

    # print(
    #     "NaN strategy returns:",
    #     oos_data["strategy_return"].isna().sum()
    # )

    # print(
    #     "NaN portfolio values:",
    #     oos_data["portfolio_value"].isna().sum()
    # )

    # print(
    #     "\nRows with NaN portfolio value:"
    # )

    # print(
    #     oos_data[
    #         oos_data["portfolio_value"].isna()
    #     ][
    #         [
    #             "timestamp",
    #             "strategy_return",
    #             "position",
    #             "portfolio_value",
    #         ]
    #     ].to_string(index=False)
    # )
    
    #--------------------------------------------------------

    trades = extract_trades(oos_data)

    if trades.empty:
        print("No trades found.")
        return
    
    winning_trades = trades[
        trades["trade_return"] > 0
    ]

    losing_trades = trades[
        trades["trade_return"] < 0
    ]

    win_rate = (
        len(winning_trades)
        / len(trades)
    )

    average_winner = (
        winning_trades["trade_return"].mean()
    )

    average_loser = (
        losing_trades["trade_return"].mean()
    )

    best_trade = (
        trades["trade_return"].max()
    )

    worst_trade = (
        trades["trade_return"].min()
    )

    average_holding_days = (
        trades["holding_days"].mean()
    )

    gross_profit = (
        winning_trades["trade_return"].sum()
    )

    gross_loss = (
        losing_trades["trade_return"].sum()
    )

    profit_factor = (
        gross_profit / abs(gross_loss)
    )
    
    total_trade_pnl = (
        trades["portfolio_pnl"].sum()
    )
    
    trades["pnl_contribution_pct"] = (
        trades["portfolio_pnl"]
        / total_trade_pnl
    )
    
    winning_pnl = trades[
        trades["portfolio_pnl"] > 0
    ]["portfolio_pnl"]

    top_5_winners = (
        winning_pnl
        .sort_values(ascending=False)
        .head(5)
        .sum()
    )

    total_winning_pnl = (
        winning_pnl.sum()
    )

    top_5_contribution = (
        top_5_winners
        / total_winning_pnl
    )
    
    # summarizing trades by entry regime
    entry_regime_summary = (
        trades
        .groupby("entry_regime")
        .agg(
            trades=("trade_return", "count"),
            win_rate=(
                "trade_return",
                lambda x: (x > 0).mean(),
            ),
            average_trade_return=(
                "trade_return",
                "mean",
            ),
            average_portfolio_return=(
                "portfolio_return",
                "mean",
            ),
            total_pnl=(
                "portfolio_pnl",
                "sum",
            ),
            average_holding_days=(
                "holding_days",
                "mean",
            ),
        )
        .sort_values(
            "total_pnl",
            ascending=False,
        )
    )
    
    # summarizing trades by average regime during the trade, which is the most common regime during the trade
    average_regime_summary = (
        trades
        .groupby("average_regime")
        .agg(
            trades=("trade_return", "count"),
            win_rate=(
                "trade_return",
                lambda x: (x > 0).mean(),
            ),
            average_trade_return=(
                "trade_return",
                "mean",
            ),
            average_portfolio_return=(
                "portfolio_return",
                "mean",
            ),
            total_pnl=(
                "portfolio_pnl",
                "sum",
            ),
            average_holding_days=(
                "holding_days",
                "mean",
            ),
        )
        .sort_values(
            "total_pnl",
            ascending=False,
        )
    )
    
    #--------------------------------------------------------
    print(
        "\n========== ENTRY REGIME SUMMARY =========="
    )

    print(
        entry_regime_summary.to_string(
            formatters={
                "win_rate": "{:.2%}".format,
                "average_trade_return": "{:.2%}".format,
                "average_portfolio_return": "{:.2%}".format,
                "total_pnl": "${:,.2f}".format,
                "average_holding_days": "{:.1f}".format,
            }
        )
    )
    
    print(
        "\n========== DOMINANT REGIME SUMMARY =========="
    )

    print(
        average_regime_summary.to_string(
            formatters={
                "win_rate": "{:.2%}".format,
                "average_trade_return": "{:.2%}".format,
                "average_portfolio_return": "{:.2%}".format,
                "total_pnl": "${:,.2f}".format,
                "average_holding_days": "{:.1f}".format,
            }
        )
    )
    
    print(
        "\nEntry regime P&L total: "
        f"${entry_regime_summary['total_pnl'].sum():,.2f}"
    )

    print(
        "Dominant regime P&L total: "
        f"${average_regime_summary['total_pnl'].sum():,.2f}"
    )
    
    
    #--------------------------------------------------------
    print("\n========== TRADE ANALYSIS ==========")

    print(
        f"Number of trades: "
        f"{len(trades)}"
    )

    print(
        f"Winning trades: "
        f"{(trades['trade_return'] > 0).sum()}"
    )

    print(
        f"Losing trades: "
        f"{(trades['trade_return'] < 0).sum()}"
    )
    
    print(
        f"Win rate: "
        f"{win_rate:.2%}"
    )

    print(
        f"Average winning trade: "
        f"{average_winner:.2%}"
    )

    print(
        f"Average losing trade: "
        f"{average_loser:.2%}"
    )

    print(
        f"Best trade: "
        f"{best_trade:.2%}"
    )

    print(
        f"Worst trade: "
        f"{worst_trade:.2%}"
    )

    print(
        f"Average holding period: "
        f"{average_holding_days:.1f} days"
    )

    print(
        f"Approximate profit factor: "
        f"{profit_factor:.2f}"
    )


    print("\n========== TRADE TABLE ==========")

    print(
        trades.to_string(index=False)
    )
    
    print("\n========== TRADE P&L SUMMARY ==========")
    
    print(
        f"Total trade P&L: "
        f"${total_trade_pnl:,.2f}"
    )

    print(
        f"Top 5 winners gross P&L: "
        f"${top_5_winners:,.2f}"
    )

    print(
        f"Total winning trade P&L: "
        f"${total_winning_pnl:,.2f}"
    )

    print(
        f"Top 5 winners contribution "
        f"to gross profit: "
        f"{top_5_contribution:.2%}"
    )
    
    


if __name__ == "__main__":
    main()
