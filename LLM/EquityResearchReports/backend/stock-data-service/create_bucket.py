# create_bucket.py
from influxdb_client import InfluxDBClient, BucketsApi

# Your setup
org = "Telluride Solutions"
token = "T7B_OVPPwqwfASNP_SzR9jbnRbdTI381GsjCn14p1r2geDrPk8UiB7LJ9SEA5eGBbuvZUMOP0hU8lBUJtwk3tA=="
url = "http://localhost:8086"
bucket_name = "stock_data"

# Connect to InfluxDB
client = InfluxDBClient(url=url, token=token, org=org)
buckets_api = client.buckets_api()

# Create bucket if not exists
buckets = buckets_api.find_buckets().buckets
if any(b.name == bucket_name for b in buckets):
    print(f"✅ Bucket '{bucket_name}' already exists.")
else:
    buckets_api.create_bucket(bucket_name=bucket_name, org=org)
    print(f"✅ Bucket '{bucket_name}' created.")