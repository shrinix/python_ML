# app.py

from flask import Flask, render_template, request, jsonify
from influx_query import query_stock_data, get_list_of_tickers, get_ticker_date_range
import socket
import consul
import os
from flask_cors import CORS

app = Flask(__name__)

# Uncommend and Enable CORS for select routes and origins only for local testing
# This setting is not needed if CORS is handled by API-Gateway or reverse proxy in production.
# CORS(app, resources={r"/api/*": {"origins": [
#     "http://localhost:8081", "http://127.0.0.1:8081"
# ]}})

#Uncomment this if running solely as a Flask app without Docker
# @app.route("/")
# def home():
#     return render_template("index.html")

@app.route("/api/tickers")
def api_tickers():
    tickers = get_list_of_tickers()
    return jsonify({"tickers": tickers})

@app.route("/api/date_range")
def api_date_range():
    tickers = get_list_of_tickers()
    min_dates, max_dates = [], []
    for t in tickers:
        min_d, max_d = get_ticker_date_range(t)
        if min_d and max_d:
            min_dates.append(min_d)
            max_dates.append(max_d)
    if not min_dates or not max_dates:
        return jsonify({"min_date": None, "max_date": None})
    min_date = min(min_dates).strftime("%Y-%m-%d")
    max_date = max(max_dates).strftime("%Y-%m-%d")
    return jsonify({"min_date": min_date, "max_date": max_date})

@app.route("/api/query", methods=["POST"])
def query():
    # Use request.get_json(force=True) for more robust parsing
    data = request.get_json(force=True, silent=True)
    print("Received data:", data)  # Debug: See the raw data

    ticker = data.get("ticker") if data else None
    start = data.get("start_date") if data else None
    end = data.get("end_date") if data else None

    print(f"Received query for ticker: {ticker}, start: {start}, end: {end}")

    if not ticker or not start or not end:
        return jsonify({"status": "error", "message": "Missing parameters"}), 400

    df = query_stock_data(ticker, start, end)
    if df.empty:
        return jsonify({"status": "empty", "data": []})
    
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return jsonify({"status": "ok", "data": df.to_dict(orient="records")})

# For local runs, set CONSUL_HOST=localhost (or leave unset for default).
# For Docker Compose, set CONSUL_HOST=consul in your service’s environment.
def register_with_consul():
    consul_host = os.environ.get("CONSUL_HOST", "localhost")
    consul_port = int(os.environ.get("CONSUL_PORT", 8500))
    service_name = "stock-viewer-service"
    service_port = int(os.environ.get("SERVICE_PORT", 5000))
    service_id = f"{service_name}-{service_port}"
    service_address = socket.gethostbyname(socket.gethostname())
    c = consul.Consul(host=consul_host, port=consul_port)
    c.agent.service.register(
        name=service_name,
        service_id=service_id,
        address=service_address,
        port=service_port,
        check=consul.Check.http(f"http://{service_address}:{service_port}/health", "10s")
    )

@app.route("/health")
def health_check():
    return jsonify({"status": "ok"}), 200
@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Not Found"}), 404

if __name__ == "__main__":
    register_with_consul()
    #Use host="127.0.0.1" (the default) to restrict access to your machine only.
    # Set host to '0.0.0.0' to be accessible from localhost as well as other containers or host,  
    # when using in Docker/production. But ensure you have proper firewall rules, reverse proxies, 
    # or API gateways in place to control access.
    app.run(host="0.0.0.0", port=int(os.environ.get("SERVICE_PORT", 6000)), debug=True)
