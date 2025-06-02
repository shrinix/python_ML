# ingest_stock_data.py
import os
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision, DeleteApi, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

# Configuration
bucket = "stock_data"
org = "Telluride Solutions"
token = "T7B_OVPPwqwfASNP_SzR9jbnRbdTI381GsjCn14p1r2geDrPk8UiB7LJ9SEA5eGBbuvZUMOP0hU8lBUJtwk3tA=="
url = "http://localhost:8086"
csv_folder = "/Users/shriniwasiyengar/git/python_ML/LLM/EquityResearchReports/backend/ai-service/source-files"

# InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()

def record_exists_old(ticker, date):
    date_obj = pd.to_datetime(date)
    start = date_obj.strftime('%Y-%m-%dT00:00:00Z')
    end = (date_obj + pd.Timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
    flux = f'''
    from(bucket: "{bucket}")
      |> range(start: {start}, stop: {end})
      |> filter(fn: (r) => r["_measurement"] == "stock_price" and r["ticker"] == "{ticker}")
      |> limit(n:1)
    '''
    result = query_api.query(flux)
    return len(result) > 0

def ingest_csv_old(file_path):
    print(f"\n📄 Processing {file_path}")
    df = pd.read_csv(file_path, header=0, infer_datetime_format=True)
    
    for _, row in df.iterrows():
        # Extract ticker from filename
        filename = os.path.basename(file_path)
        ticker = filename.split("_")[2]
        ts = row["Date"]

        if record_exists_old(ticker, ts):
            print(f"⚠️  Duplicate for {ticker} on {pd.to_datetime(ts).date()}, skipping.")
            continue

        point = (
            Point("stock_price")
            .tag("ticker", ticker)
            .field("Open", float(row["Open"]))
            .field("High", float(row["High"]))
            .field("Low", float(row["Low"]))
            .field("Adj Close", float(row["Adj Close"]))
            .field("Close", float(row["Close"]))
            .field("Volume", int(row["Volume"]))
            .time(ts, WritePrecision.NS)
        )

        write_api.write(bucket=bucket, org=org, record=point)
        print(f"✅ Inserted {ticker} on {pd.to_datetime(ts).date()}")

def normalize_to_midnight_utc(ts):
    """Ensure timestamp is at midnight UTC."""
    dt = pd.to_datetime(ts).tz_convert("UTC") if pd.to_datetime(ts).tzinfo else pd.to_datetime(ts).tz_localize("UTC")
    return dt.replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0)

def record_exists_exact(ticker, ts):
    """Check if a record exists for the ticker at the exact timestamp."""
    ts_str = ts.strftime('%Y-%m-%dT%H:%M:%SZ')
    flux = f'''
    from(bucket: "{bucket}")
      |> range(start: {ts_str}, stop: {(ts + pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')})
      |> filter(fn: (r) => r["_measurement"] == "stock_price" and r["ticker"] == "{ticker}" and r._time == {ts_str})
      |> limit(n:1)
    '''
    result = query_api.query(flux)
    return any(table.records for table in result)

def deduplicate_df(df):
    """Drop duplicate dates, keeping the first occurrence."""
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize("UTC", nonexistent='NaT', ambiguous='NaT')
    df["Date"] = df["Date"].dt.normalize()  # sets time to 00:00:00
    return df.drop_duplicates(subset=["Date"])

def record_exists_by_date(ticker, date):
    """Check if any record exists for the ticker on the given date (ignores time)."""
    date_obj = pd.to_datetime(date)
    start = date_obj.strftime('%Y-%m-%dT00:00:00Z')
    end = (date_obj + pd.Timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
    flux = f'''
    from(bucket: "{bucket}")
      |> range(start: {start}, stop: {end})
      |> filter(fn: (r) => r["_measurement"] == "stock_price" and r["ticker"] == "{ticker}")
      |> limit(n:1)
    '''
    result = query_api.query(flux)
    return any(table.records for table in result)

def normalize_to_midnight_utc(ts):
    """Ensure timestamp is at midnight UTC, no microseconds/nanoseconds."""
    dt = pd.to_datetime(ts)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    # Remove microseconds and nanoseconds
    return dt.replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0)

def del_all_records():

    #check with user before deleting
    confirm = input("Are you sure you want to delete all existing stock data in InfluxDB? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ Operation cancelled. No data was deleted.")
        return
    print("🗑️  Deleting all existing stock data in InfluxDB...")
    # Delete the bucket
    bucket_obj = buckets_api.find_bucket_by_name(bucket)
    if bucket_obj:
        buckets_api.delete_bucket(bucket_obj)

    # Recreate the bucket
    buckets_api.create_bucket(bucket_name=bucket, org=org)

    print("🗑️  Deleted all existing stock data in InfluxDB"
          " (this is a one-time operation, remove this line in production).")

    #count records to verify deletion
    query = f'''
    from(bucket: "{bucket}")
    |> range(start: {start}, stop: {stop})
    |> filter(fn: (r) => r["_measurement"] == "stock_price")
    |> keep(columns: ["_time"])
    |> group()
    |> distinct(column: "_time")
    '''

    result = query_api.query(query)
    count = sum(len(table.records) for table in result)
    print(f"📊 Total records after deletion: {count}"
        " (should be 0 if deletion was successful).")

    input("Press Enter to continue...")

def ingest_csv(file_path):
    print(f"\n📄 Processing {file_path}")
    df = pd.read_csv(file_path, header=0, infer_datetime_format=True)

    # Normalize and deduplicate by date (midnight UTC)
    df["Date"] = pd.to_datetime(df["Date"])
    # Only localize if not already tz-aware
    if df["Date"].dt.tz is None:
        df["Date"] = df["Date"].dt.tz_localize("UTC", nonexistent='NaT', ambiguous='NaT')
    else:
        df["Date"] = df["Date"].dt.tz_convert("UTC")
    df["Date"] = df["Date"].dt.normalize()
    df = df.drop_duplicates(subset=["Date"])

    # Extract ticker from filename
    filename = os.path.basename(file_path)
    ticker = filename.split("_")[2]

    # Optional: Print unique dates for verification
    print(f"Unique dates to ingest for {ticker}: {df['Date'].nunique()}")

    records_written_count = 0
    # Ingest only one record per ticker per day
    for _, row in df.iterrows():
        ts = normalize_to_midnight_utc(row["Date"])

        # Check for duplicates in InfluxDB before writing
        if record_exists_by_date(ticker, ts):
            print(f"⚠️  Duplicate for {ticker} on {ts.date()}, skipping.")
            continue

        point = (
            Point("stock_price")
            .tag("ticker", ticker)
            .field("Open", float(row["Open"]))
            .field("High", float(row["High"]))
            .field("Low", float(row["Low"]))
            .field("Adj Close", float(row["Adj Close"]))
            .field("Close", float(row["Close"]))
            .field("Volume", int(row["Volume"]))
            .time(ts, WritePrecision.NS)
        )

        write_api.write(bucket=bucket, org=org, record=point)
        records_written_count += 1
        print(f"✅ Inserted {ticker} on {ts.date()}")

    print(f"📊 Total records written for {ticker}: {records_written_count}"
          " (should be 1 per day if no duplicates).")
# Increase the timeout for delete operations
start = "1970-01-01T00:00:00Z"
stop = "2100-01-01T00:00:00Z"
buckets_api = client.buckets_api()

#del_all_records()

# Scan folder
for file in os.listdir(csv_folder):
    if file.endswith(".csv") and "_all_" not in file:
        ingest_csv(os.path.join(csv_folder, file))

client.close()