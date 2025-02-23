from flask import Flask, jsonify
import pandas as pd
import portfolio_optimizer  # Import the data processing module
from datetime import date, timedelta
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

#post method to trigger the portfolio analysis calculations
@app.route('/api/portfolio-analysis', methods=['POST'])
def portfolio_analysis():
    print ("Invoking portfolio analysis")
    load_data = False
    portfolio_weights_df, portfolio_value_df, portfolio_weights_delta = portfolio_optimizer.run_portfolio_analysis(load_data)
    return jsonify({'message': 'Portfolio analysis completed'})

@app.route('/api/ticker-plot-data', methods=['GET'])
def get_timeseries_plot_data():
    try:
        output_files_folder = os.environ.get('output_files_folder')
        merged_stock_data = pd.read_csv(f'{output_files_folder}/merged_stock_data.csv')
        print(f"Retreived {merged_stock_data.shape[0]} rows of merged_stock_data")

        #check if the data is empty
        if merged_stock_data.empty:
            return jsonify([])
                #re
        print(merged_stock_data.head())
        # Convert the data to a JSON object
        data_dict = merged_stock_data.to_dict(orient='records')    
        return jsonify(data_dict)
    except Exception as e:
        return jsonify({'message': 'Error fetching time series plot data', 'error': str(e)}), 500

@app.route('/api/portfolio-plot-data', methods=['GET'])
def get_portfolio_plot_data():
    
    output_files_folder = os.environ.get('output_files_folder')
    load_data = False
    portfolio_weights_df, portfolio_value_df, portfolio_weights_delta = portfolio_optimizer.run_portfolio_analysis(load_data)
    print(f"Retreived {portfolio_value_df.shape[0]} rows of portfolio_value_df")
    print(f"Retreived {portfolio_weights_df.shape[0]} rows of portfolio_weights_df")
    print(f"Retreived {portfolio_weights_delta.shape[0]} rows of portfolio_weights_delta")

    data_dict = {
        'portfolio_weights': portfolio_weights_df.to_dict(orient='records'),
        'portfolio_value': portfolio_value_df.to_dict(orient='records'),
        'portfolio_weights_delta': portfolio_weights_delta.to_dict(orient='records')
    }
    # # Convert the data to a dictionary
    # data_dict = data.to_dict(orient='records')
    
    return jsonify(data_dict)

#add an initialization method to load configuration params and data.
#this method should be called when the flask app is started and should not be accessible via the API
def init():
    # Load the configuration parameters
    portfolio_optimizer.load_configuration_params()
    load_from_files = True
    portfolio_optimizer.prepare_data(load_from_files)

if __name__ == '__main__':
    init()
    app.run(debug=True)