import duckdb
from dotenv import load_dotenv

load_dotenv()
try:
    conn = duckdb.connect('md:kalshi_trading?motherduck_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Ind5bm5vc2hlcmlkYW5AZ21haWwuY29tIiwibWRSZWdpb24iOiJhd3MtdXMtZWFzdC0xIiwic2Vzc2lvbiI6Ind5bm5vc2hlcmlkYW4uZ21haWwuY29tIiwicGF0IjoidDdpaEhuMzdPcGw2dkpzT0tDa3EzSnpES0cxdmZwODBKSEl4S3M3V1ZQZyIsInVzZXJJZCI6ImI1Y2NjY2NhLTZiZmItNDkxNi1hMWNhLTFkZTAwYjcyNjJmOCIsImlzcyI6Im1kX3BhdCIsInJlYWRPbmx5IjpmYWxzZSwidG9rZW5UeXBlIjoicmVhZF93cml0ZSIsImlhdCI6MTc3NDQ1NzAyOX0.suLFxm77YYWc7IICu5RTA5SxxtVG4ZwfEeVO5lFHUd8')
    
    print("\n--- TABLES IN kalshi_trading ---")
    print(conn.execute("SHOW TABLES").df())

    print("\n--- DESCRIBE main.kalshi_ticks ---")
    print(conn.execute("DESCRIBE main.kalshi_ticks").df())

    print("\n--- COUNT ---")
    print(conn.execute("SELECT COUNT(*) FROM main.kalshi_ticks").df())
    
    print("\n--- SAMPLE ROW ---")
    print(conn.execute("SELECT * FROM main.kalshi_ticks LIMIT 1").df())
except Exception as e:
    print(f"Error connecting: {e}")
