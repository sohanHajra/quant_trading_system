import pandas as pd
from src.research.walk_forward import (
    load_market_data,
    run_walk_forward,
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

    in_trade = False
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
            entry_date = data.loc[i, "timestamp"]
            entry_price = price

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        elif in_trade and position == 0:

            exit_date = data.loc[i, "timestamp"]
            exit_price = price

            trade_return = (
                exit_price / entry_price
            ) - 1

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
                    "holding_days": holding_days,
                    "exit_reason": "signal",
                }
            )

            in_trade = False
            entry_date = None
            entry_price = None

    # --------------------------------------------------------
    # Handle an open position at the end of the OOS sample
    # --------------------------------------------------------

    if in_trade:

        exit_date = data.iloc[-1]["timestamp"]
        exit_price = data.iloc[-1]["close"]

        trade_return = (
            exit_price / entry_price
        ) - 1

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
                "holding_days": holding_days,
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


if __name__ == "__main__":
    main()
