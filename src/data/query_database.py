import duckdb


DATABASE_PATH = "data/market.duckdb"


con = duckdb.connect(DATABASE_PATH)

result = con.execute(
    """
    SELECT
        COUNT(*) AS rows,
        MIN(timestamp) AS first_date,
        MAX(timestamp) AS last_date
    FROM prices
    """
).fetchdf()

print(result)

con.close()
