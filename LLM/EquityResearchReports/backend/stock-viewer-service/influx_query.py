# influx_query.py

import pandas as pd
from influxdb_client import InfluxDBClient

# Config
bucket = "stock_data"
org = "Telluride Solutions"
token = "T7B_OVPPwqwfASNP_SzR9jbnRbdTI381GsjCn14p1r2geDrPk8UiB7LJ9SEA5eGBbuvZUMOP0hU8lBUJtwk3tA=="
url = "http://localhost:8086"

def query_stock_data(ticker, start_date, end_date):
	client = InfluxDBClient(url=url, token=token, org=org)
	query_api = client.query_api()

	# Ensure start_date and end_date are RFC3339 timestamps (no extra quotes in Flux)
	if hasattr(start_date, 'strftime'):
		start_rfc3339 = start_date.strftime('%Y-%m-%dT00:00:00Z')
	elif "T" in str(start_date):
		start_rfc3339 = str(start_date)
	else:
		start_rfc3339 = f"{start_date}T00:00:00Z"

	if hasattr(end_date, 'strftime'):
		end_rfc3339 = end_date.strftime('%Y-%m-%dT23:59:59Z')
	elif "T" in str(end_date):
		end_rfc3339 = str(end_date)
	else:
		end_rfc3339 = f"{end_date}T23:59:59Z"

	print(f"Querying data for ticker: {ticker}, from {start_rfc3339} to {end_rfc3339}")  # Debug

	flux = f'''
	from(bucket: "{bucket}")
	|> range(start: {start_rfc3339}, stop: {end_rfc3339})
	|> filter(fn: (r) => r["_measurement"] == "stock_price" and r["ticker"] == "{ticker}")
	|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
	|> sort(columns: ["_time"])
	'''
	print("Flux query:\n", flux)  # Debug: See the actual query

	records = []
	tables = query_api.query(flux)
	for table in tables:
		for rec in table.records:
			records.append({
				"Date": pd.to_datetime(rec["_time"]),
				"Open": rec["Open"],
				"High": rec["High"],
				"Low": rec["Low"],
				"Close": rec["Close"],
				"Adj Close": rec["Adj Close"],
				"Volume": int(rec["Volume"])
			})

	client.close()
	df = pd.DataFrame(records)
	numeric_cols = df.select_dtypes(include='number').columns
	df[numeric_cols] = df[numeric_cols].round(2)
	return df

def get_field_names(bucket, measurement):
	query = f'''
	from(bucket: "{bucket}")
	  |> range(start: -30d)
	  |> filter(fn: (r) => r._measurement == "{measurement}")
	  |> keep(columns: ["_field"])
	  |> distinct(column: "_field")
	'''

	with InfluxDBClient(url=url, token=token, org=org) as client:
		query_api = client.query_api()
		result = query_api.query(query)
		
		# Extract unique _field values
		fields = set()
		for table in result:
			for record in table.records:
				fields.add(record.get_value())

		return list(fields)
	
#Function to get schema of the database
def get_database_schema():
	client = InfluxDBClient(url=url, token=token, org=org)
	query_api = client.query_api()

	# Get all measurements in the bucket
	flux_measurements = f'''
	import "influxdata/influxdb/schema"
	schema.measurements(bucket: "{bucket}")
	'''
	tables = query_api.query(flux_measurements)
	measurements = [rec.values["_value"] for table in tables for rec in table.records]

	schema = []
	# For each measurement, get its fields and types
	for measurement in measurements:
		
		try:
			fields = get_field_names(bucket, measurement)
			if fields:
				schema.append({
					"measurement": measurement,
					"fields": fields
				})
			else:
				print(f"No fields found for measurement '{measurement}'")
		except ValueError as ve:
			print(f"Skipping measurement '{measurement}' due to ValueError: {ve}")
		except Exception as e:
			print(f"Error processing measurement '{measurement}': {e}")

		#get tag keys:
		query = f'''
			import "influxdata/influxdb/schema"

			schema.tagKeys(
			bucket: "{bucket}",
			predicate: (r) => r._measurement == "{measurement}",
			start: -10d
			)
			'''

		# with InfluxDBClient(url=url, token=token, org=org) as client:
		# 	query_api = client.query_api()
		# 	result = query_api.query(query)

		# 	tag_keys = [record.get_value() for table in result for record in table.records]
		# 	print(f"Tag keys for measurement '{measurement}': {tag_keys}")  # Debug
		# 	if tag_keys:
		# 		schema[-1]["tag_keys"] = tag_keys
		# 	else:
		# 		schema[-1]["tag_keys"] = []
	client.close()
	return schema
#Function to test the data in the database
#Find the earliest and latest dates for each ticker in the database. Find the number of records between these dates. This number
#should match the number of business days between the two dates as per the US stock market calendar.
def get_ticker_date_range(ticker):
	client = InfluxDBClient(url=url, token=token, org=org)
	query_api = client.query_api()

	# Query all records for the ticker (adjust range as needed)
	flux = f'''
	from(bucket: "{bucket}")
	  |> range(start: 0)
	  |> filter(fn: (r) => r["_measurement"] == "stock_price" and r["ticker"] == "{ticker}")
	  |> sort(columns: ["_time"])
	  |> keep(columns: ["_time"])
	'''

	tables = query_api.query(flux)
	times = []
	for table in tables:
		for rec in table.records:
			# _time is always present in InfluxDB records
			times.append(pd.to_datetime(rec["_time"]))

	client.close()
	if not times:
		return None, None
	return min(times), max(times)

def get_ticker_record_count_v2(ticker, start_date, end_date):
	client = InfluxDBClient(url=url, token=token, org=org)
	query_api = client.query_api()

	# Ensure start_date and end_date are RFC3339 strings
	if hasattr(start_date, 'strftime'):
		start_rfc3339 = start_date.strftime('%Y-%m-%dT00:00:00Z')
	elif "T" in str(start_date):
		start_rfc3339 = str(start_date)
	else:
		start_rfc3339 = f"{start_date}T00:00:00Z"

	if hasattr(end_date, 'strftime'):
		end_rfc3339 = end_date.strftime('%Y-%m-%dT23:59:59Z')
	elif "T" in str(end_date):
		end_rfc3339 = str(end_date)
	else:
		end_rfc3339 = f"{end_date}T23:59:59Z"

	flux = f"""
	from(bucket: "{bucket}")
	|> range(start: -30d)
	|> filter(fn: (r) => r._measurement == "stock_price" and r["ticker"] == "{ticker}")
	|> filter(fn: (r) => r._time >= {start_rfc3339} and r._time <= {end_rfc3339})
	|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
	|> group(columns: ["_time"])
	|> map(fn: (r) => ({{ r with count: 1 }}))
	|> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
	"""
	
	results = query_api.query(flux)

	daily_counts = {}
	for table in results:
		for record in table.records:
			date_str = record.get_time().date().isoformat()
			count = record.get_value()
			daily_counts[date_str] = count

	client.close()
	return daily_counts

def get_ticker_record_count(ticker, start_date, end_date):
	client = InfluxDBClient(url=url, token=token, org=org)
	query_api = client.query_api()

	# Ensure start_date and end_date are RFC3339 strings
	if hasattr(start_date, 'strftime'):
		start_rfc3339 = start_date.strftime('%Y-%m-%dT00:00:00Z')
	elif "T" in str(start_date):
		start_rfc3339 = str(start_date)
	else:
		start_rfc3339 = f"{start_date}T00:00:00Z"

	if hasattr(end_date, 'strftime'):
		end_rfc3339 = end_date.strftime('%Y-%m-%dT23:59:59Z')
	elif "T" in str(end_date):
		end_rfc3339 = str(end_date)
	else:
		end_rfc3339 = f"{end_date}T23:59:59Z"

	flux = f'''
	from(bucket: "{bucket}")
	  |> range(start: {start_rfc3339}, stop: {end_rfc3339})
	  |> filter(fn: (r) => r["_measurement"] == "stock_price" and r["ticker"] == "{ticker}")
	  |> keep(columns: ["_time"])
	  |> group()
	  |> distinct(column: "_time")
	'''

	tables = query_api.query(flux)
	count = 0
	for table in tables:
		for rec in table.records:
			count += 1

	client.close()
	return count

def get_ticker_business_days(ticker, start_date, end_date):
	from pandas.tseries.offsets import BDay

	# Convert to datetime if not already
	start_date = pd.to_datetime(start_date)
	end_date = pd.to_datetime(end_date)

	#use US calendar for business days
	from pandas.tseries.holiday import USFederalHolidayCalendar
	from pandas.tseries.offsets import CustomBusinessDay
	# Create a custom business day offset using US Federal Holidays
	us_bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())

	# Generate business days between the two dates
	business_days = pd.date_range(start=start_date, end=end_date, freq=us_bday)
	
	return len(business_days)

def get_ticker_data_summary(ticker):
	min_date, max_date = get_ticker_date_range(ticker)
	if not min_date or not max_date:
		return {"status": "error", "message": "No data found for ticker"}

	start_date = min_date.strftime('%Y-%m-%dT00:00:00Z')
	end_date = max_date.strftime('%Y-%m-%dT23:59:59Z')

	record_count = get_ticker_record_count(ticker, start_date, end_date)
	business_days = get_ticker_business_days(ticker, min_date, max_date)
	total_days = (max_date - min_date).days + 1  # Include both start and end dates

	return {
		"ticker": ticker,
		"start_date": min_date.strftime('%Y-%m-%d'),
		"end_date": max_date.strftime('%Y-%m-%d'),
		"record_count": record_count,
		"business_days": business_days,
		"total_days": total_days,
	}

def get_list_of_tickers():
	client = InfluxDBClient(url=url, token=token, org=org)
	query_api = client.query_api()

	flux = f'''
	from(bucket: "{bucket}")
	|> range(start: 0)
	|> filter(fn: (r) => r["_measurement"] == "stock_price")
	|> keep(columns: ["ticker"])
	|> group(columns: ["ticker"])
	|> distinct(column: "ticker")
	'''

	tables = query_api.query(flux)
	tickers = set()
	for table in tables:
		for rec in table.records:
			tickers.add(rec["ticker"])

	client.close()
	return list(tickers)

def check_intraday_records():
	client = InfluxDBClient(url=url, token=token, org=org)
	query_api = client.query_api()

	flux = f'''
	import "date"
	from(bucket: "{bucket}")
	|> range(start: 0)
	|> filter(fn: (r) => r["_measurement"] == "stock_price" and r["ticker"] == "{ticker}")
	|> filter(fn: (r) => date.hour(t: r._time) != 0 or date.minute(t: r._time) != 0 or date.second(t: r._time) != 0)
	|> keep(columns: ["_time"])
		'''
	tables = query_api.query(flux)
	times = []
	for table in tables:
		for rec in table.records:
			times.append(pd.to_datetime(rec["_time"]))

	client.close()
	if not times:
		print("No intraday records found for the specified ticker.")
		return 0

def count_duplicate_dates(ticker):
    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()
    flux = f'''
    from(bucket: "{bucket}")
    |> range(start: 0)
    |> filter(fn: (r) => r["_measurement"] == "stock_price" and r["ticker"] == "{ticker}")
    |> keep(columns: ["_time"])
    |> group()
    |> distinct(column: "_time")
    '''
    tables = query_api.query(flux)
    times = []
    for table in tables:
        for rec in table.records:
            # After distinct, the unique _time is in _value
            times.append(pd.to_datetime(rec.values["_value"]).date())

    client.close()
    # Count occurrences of each date using a dictionary
    date_counts = {}
    for d in times:
        date_counts[d] = date_counts.get(d, 0) + 1

    # Find dates with more than one unique timestamp (should be 1 per day)
    duplicate_dates = {d: c for d, c in date_counts.items() if c > 1}
    duplicate_count = len(duplicate_dates)
    total_duplicates = sum(c - 1 for c in duplicate_dates.values())
    print(f"Days with duplicate records: {duplicate_count}")
    print(f"Total extra records (above 1 per day): {total_duplicates}")

    # Print all records for one duplicate date, if any
    if duplicate_count > 0:
        duplicate_date = next(iter(duplicate_dates))
        print(f"Duplicate date found: {duplicate_date}")
        start_rfc3339 = f"{duplicate_date}T00:00:00Z"
        end_rfc3339 = (pd.to_datetime(duplicate_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
        flux = f'''
        from(bucket: "{bucket}")
        |> range(start: {start_rfc3339}, stop: {end_rfc3339})
        |> filter(fn: (r) => r["_measurement"] == "stock_price" and r["ticker"] == "{ticker}")
        |> sort(columns: ["_time"])
        '''
        tables = query_api.query(flux)
        records_for_date = []
        for table in tables:
            for rec in table.records:
                records_for_date.append(rec.values)
        print(f"All records for {ticker} on {duplicate_date}:")
        for rec in records_for_date:
            print(rec)

    print(f"Duplicate records for {ticker}: {duplicate_count}, Total duplicates: {total_duplicates}")
    return duplicate_count, total_duplicates

if __name__ == "__main__":

	schema = get_database_schema()
	print("Database Schema:", schema)

	tickers = get_list_of_tickers()
	print("Available tickers:", tickers)
	# Example usage
	ticker = "AAPL"

	for t in tickers:
		
		start_date, end_date = get_ticker_date_range(t)
		print(f"Ticker: {t}, Start Date: {start_date}, End Date: {end_date}")

		df = query_stock_data(ticker, start_date, end_date)
		print(df.head())

		summary = get_ticker_data_summary(ticker)
		print(summary)
		count_duplicate_dates(ticker)

	check_intraday_records()

	