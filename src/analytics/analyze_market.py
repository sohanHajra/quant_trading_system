import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FILE_PATH = "data/raw/SPY_daily.csv"


# Load data
df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)

# Make sure data is sorted chronologically
df = df.sort_index()

print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())


df["return"] = df["close"].pct_change()

print("\nDaily returns:")
print(df["return"].tail())


daily_volatility = df["return"].std()

print(f"\nDaily volatility: {daily_volatility:.4%}")


annualized_volatility = daily_volatility * np.sqrt(252)

print(f"Annualized volatility: {annualized_volatility:.2%}")


df["cumulative_return"] = (1 + df["return"]).cumprod()

print("\nCumulative return:")
print(df["cumulative_return"].tail())


df["wealth"] = df["cumulative_return"]

running_max = df["wealth"].cummax()

df["drawdown"] = df["wealth"] / running_max - 1

max_drawdown = df["drawdown"].min()

print(f"\nMaximum drawdown: {max_drawdown:.2%}")


plt.figure(figsize=(12, 6))

plt.plot(df.index, df["close"])

plt.title("SPY Price")
plt.xlabel("Date")
plt.ylabel("Price")

plt.grid(True)
plt.tight_layout()

plt.show()


plt.figure(figsize=(12, 4))

plt.plot(df.index, df["drawdown"])

plt.title("SPY Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")

plt.grid(True)
plt.tight_layout()

plt.show()


total_return = df["cumulative_return"].iloc[-1] - 1

print("\n========== MARKET SUMMARY ==========")

print(f"Total return:          {total_return:.2%}")
print(f"Daily volatility:      {daily_volatility:.2%}")
print(f"Annualized volatility: {annualized_volatility:.2%}")
print(f"Maximum drawdown:       {max_drawdown:.2%}")
