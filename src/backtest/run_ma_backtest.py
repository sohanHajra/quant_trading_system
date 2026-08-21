import duckdb
import matplotlib.pyplot as plt

from src.backtest.simple_backtest import run_backtest
from src.analytics.performance import calculate_metrics


DATABASE_PATH = "data/market.duckdb"


con = duckdb.connect(DATABASE_PATH)

df = con.execute(
    """
    SELECT *
    FROM prices
    ORDER BY timestamp
    """
).fetchdf()

con.close()


results = run_backtest(df)

print("\nLast 10 rows:")
print(
    results[
        [
            "timestamp",
            "close",
            "sma_fast",
            "sma_slow",
            "signal",
            "position",
            "portfolio_value",
        ]
    ].tail(10)
)

results["buy_hold"] = (
    100_000 *
    (
        results["close"] /
        results["close"].iloc[0]
    )
)

final_value = results["portfolio_value"].iloc[-1]

total_return = (
    final_value / 100_000
) - 1

metrics = calculate_metrics(results)
total_transaction_cost = (
    results["transaction_cost"].sum()
)

print(
    f"Total transaction costs: "
    f"{total_transaction_cost:.4%}"
)


print("\n========== PERFORMANCE REPORT ==========")

print(
    f"Final value:          ${metrics['final_value']:,.2f}"
)

print(
    f"Total return:         {metrics['total_return']:.2%}"
)

print(
    f"Annualized return:    {metrics['annualized_return']:.2%}"
)

print(
    f"Annualized volatility:{metrics['annualized_volatility']:.2%}"
)

print(
    f"Sharpe ratio:         {metrics['sharpe_ratio']:.2f}"
)

print(
    f"Maximum drawdown:     {metrics['max_drawdown']:.2%}"
)

print(
    f"Trade count:          {metrics['trade_count']}"
)

# initial report
# print("\n========== BACKTEST ==========")

# print(
#     f"Initial capital: ${100_000:,.2f}"
# )

# print(
#     f"Final capital:   ${final_value:,.2f}"
# )

# print(
#     f"Total return:    {total_return:.2%}"
# )


plt.figure(figsize=(12, 6))

plt.plot(
    results["timestamp"],
    results["portfolio_value"],
    label="MA Strategy",
)

plt.plot(
    results["timestamp"],
    results["buy_hold"],
    label="Buy & Hold",
)


plt.title("20/50 Moving Average Strategy")

plt.xlabel("Date")
plt.ylabel("Portfolio Value")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
