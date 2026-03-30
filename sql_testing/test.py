import duckdb

# Path to your file
file_path = 'data_w/raw/possessions_202526.parquet'

# Query the first row
# DuckDB can treat the file path string directly as a table name
# row = duckdb.query(f"SELECT * FROM '{file_path}' LIMIT 1").fetchone()

# print(row)
# This converts the query result into a Pandas DataFrame
df = duckdb.query(f"SELECT * FROM '{file_path}' LIMIT 1").df()

#print(df)
print(list(df.columns))