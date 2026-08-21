import duckdb
import pandas as pd


DATABASE_PATH = "data/market.duckdb"


def save_prices(df: pd.DataFrame, table_name: str = "prices"):
    """Save market data to DuckDB."""

    con = duckdb.connect(DATABASE_PATH)

    con.register("price_data", df)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT *
        FROM price_data
        """
    )

    con.close()

    print(
        f"Saved {len(df)} rows to {DATABASE_PATH}"
    )
