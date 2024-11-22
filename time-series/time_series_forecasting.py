#pip install pmdarima
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_data():
    # Generate synthetic financial data with a trend
    np.random.seed(42)
    n_points = 100
    time = np.arange(n_points)
    trend = 0.5 * time + np.random.normal(scale=5, size=n_points)
    financial_data = pd.DataFrame({'Time': time, 'Trend': trend})

    # Split the data into training and testing sets
    train_size = int(0.8 * n_points)
    train_data = financial_data.head(train_size).copy()
    test_data = financial_data.tail(n_points - train_size).copy()

    return train_data, test_data

def build_and_run_regression_model(train_data, test_data):

    print("Forecasting using linear regression model...")
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(train_data[['Time']], train_data['Trend'])

    # Predict the trend for the test set
    test_data['Trend_Predicted'] = model.predict(test_data[['Time']])
    # test_data['Trend_Predicted'] = scaler_trend.inverse_transform(test_data[['Trend_Predicted']])
    
    # Calculate performance metrics
    mae = mean_absolute_error(test_data['Trend'], test_data['Trend_Predicted'])
    mse = mean_squared_error(test_data['Trend'], test_data['Trend_Predicted'])
    rmse = np.sqrt(mse)

    # Print performance metrics
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    return test_data['Trend_Predicted'], mae, mse, rmse

# #use minmaxscaler to scale the data
# from sklearn.preprocessing import MinMaxScaler
# scaler_trend = MinMaxScaler()
# financial_data['Trend'] = scaler_trend.fit_transform(financial_data[['Trend']])
# scaler_time = MinMaxScaler()
# financial_data['Time'] = scaler_time.fit_transform(financial_data[['Time']])


def build_and_run_arima_model(train_data, test_data):

    print("Forecasting using ARIMA model...")
    # Auto-fit ARIMA model
    auto_model = auto_arima(train_data['Trend'], seasonal=False, suppress_warnings=True)
    fit_model = auto_model.fit(train_data['Trend'])

    # Forecast the trend for the test set
    forecast = fit_model.predict(n_periods=len(test_data))
    # test_data['Trend_Predicted'] = scaler_trend.inverse_transform(test_data[['Trend_Predicted']])

    # Calculate performance metrics
    mae = mean_absolute_error(test_data['Trend'], forecast)
    mse = mean_squared_error(test_data['Trend'], forecast)
    rmse = np.sqrt(mse)

    # Print performance metrics
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    return forecast, mae, mse, rmse

#Create a keras tensorflow LSTM model for time series forecasting
def build_and_run_lstm_model(train_data, test_data):

    print("Forecasting using LSTM model...")
    # Define the number of time steps
    n_steps = 3

    X_train, y_train = train_data[['Time']], train_data['Trend']
    
    print("Shape of X_train: ", X_train.shape)
    print("Shape of y_train: ", y_train.shape)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the LSTM model
    model.fit(X_train, y_train, epochs=200, verbose=0)

    # Predict the trend for the test set
    test_data['Trend_Predicted'] = model.predict(test_data[['Time']])
    print("Shape of y_pred: ", test_data['Trend_Predicted'].shape)


    # Calculate performance metrics
    mae = mean_absolute_error(test_data['Trend'], test_data['Trend_Predicted'])
    mse = mean_squared_error(test_data['Trend'], test_data['Trend_Predicted'])
    rmse = np.sqrt(mse)

    # Print performance metrics
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    return test_data['Trend_Predicted'], mae, mse, rmse

def plot_results(ax, train_data, test_data, forecast, mae, mse, rmse,  model_name, plot_title):

    """Plot the actual and predicted trends."""
    # Plot the actual and predicted trends for the regression model
    ax.plot(train_data['Time'], train_data['Trend'], label='Training Data', marker='o')
    ax.plot(test_data['Time'], test_data['Trend'], label='Actual Trend', marker='o')
    label = 'Predicted Trend ({})'.format(model_name)
    ax.plot(test_data['Time'], forecast, label=label, linestyle='--', marker='o')
    ax.set_title(plot_title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()

    #print the performance data mae, mse and rmse in a small box on bottom right
    textstr = '\n'.join((
        r'MAE=%.2f' % (mae, ),
        r'MSE=%.2f' % (mse, ),
        r'RMSE=%.2f' % (rmse, )))
    
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)

    renderer = ax.figure.canvas.get_renderer()
    shift_left = 0.20
    bbox = ax.text(0.95 - shift_left , 0.05, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='bottom', bbox=props).get_window_extent(renderer=renderer)
    
    return ax
    
if __name__ == "__main__":

    # Prepare the financial data
    train_data, test_data = prepare_data()

    # Build and run the regression model
    forecast_regression, mae_regression, mse_regression, rmse_regression = build_and_run_regression_model(train_data, test_data)

    # Build and run the ARIMA model
    forecast_arima, mae_arima, mse_arima, rmse_arima = build_and_run_arima_model(train_data, test_data)

    # Prepare the financial data
    # train_data, test_data = prepare_data()

    # Build and run the LSTM model
    forecast_lstm, mae_lstm, mse_lstm, rmse_lstm = build_and_run_lstm_model(train_data, test_data)

    # Plot the results
   # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot results for Regression model
    plot_results(axes[0], train_data, test_data, forecast_regression, mae_regression, mse_regression, rmse_regression,
                 'Regression', 'Regression Model')

    # Plot results for ARIMA model
    plot_results(axes[1], train_data, test_data, forecast_arima, mae_arima, mse_arima, rmse_arima,
                 'ARIMA', 'ARIMA Model')

    # Plot results for LSTM model
    plot_results(axes[2], train_data, test_data, forecast_lstm, mae_lstm, mse_lstm, rmse_lstm, 
                 'LSTM', 'LSTM Model')

    plt.tight_layout()
    plt.show()