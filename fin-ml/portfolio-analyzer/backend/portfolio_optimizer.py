from datetime import date, timedelta
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import configparser
import os
import json
from matplotlib.backends.backend_pdf import PdfPages

def plot_time_series(axes, merged_stock_data, currency):
    
    sns.set_theme(style='whitegrid')
    for ticker in merged_stock_data['Ticker'].unique():
        ticker_data = merged_stock_data[merged_stock_data['Ticker'] == ticker]
        last_point = ticker_data.iloc[-1]
        axes.text(last_point['Start Date'], last_point['Adj Close'], ticker, fontsize=9, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    #plot the lineplot of the Adj Close price over time. Use the Ticker column to color the lines
    #Use the start_date column from timeframes_df as the x-axis
    #Use the Adj Close column from stock_data as the y-axis
    #Use the Ticker column from stock_data to color the lines
    sns.lineplot(data=merged_stock_data, x='Start Date', y='Adj Close', hue='Ticker', ax=axes)
    axes.set_title('Adj Close Price Over Time', fontsize=12)
    axes.set_xlabel('Start Date', fontsize=12)
    axes.set_ylabel(f'Adj Close Price ({currency})', fontsize=12)
    axes.legend(title='Ticker', title_fontsize='10', fontsize='8', loc='upper left')
    axes.grid(True)

    axes.set_xticklabels(axes.get_xticklabels(), rotation=45)

def plot_daily_returns(axes, merged_stock_data):

    unique_tickers = merged_stock_data['Ticker'].unique()
    for ticker in unique_tickers:
        ticker_data = merged_stock_data[merged_stock_data['Ticker'] == ticker]
        sns.histplot(ticker_data['Daily Return'].dropna(), ax=axes, bins=50, kde=True, label=ticker, alpha=0.5)

    axes.set_title('Distribution of Daily Returns', fontsize=12)
    axes.set_xlabel('Daily Return', fontsize=14)
    axes.set_ylabel('Frequency', fontsize=14)
    axes.legend(title='Ticker', title_fontsize='10', fontsize='8', loc='upper left')
    axes.grid(True)
    
def plot_moving_averages(axes, merged_stock_data, short_window, long_window):

    unique_tickers = merged_stock_data['Ticker'].unique()

    for ticker in unique_tickers:
        ticker_data = merged_stock_data[merged_stock_data['Ticker'] == ticker].copy()
        ticker_data['50_MA'] = ticker_data['Adj Close'].rolling(window=short_window).mean()
        ticker_data['200_MA'] = ticker_data['Adj Close'].rolling(window=long_window).mean()

        axes.plot(ticker_data['Start Date'], ticker_data['Adj Close'], label=f'{ticker} Adj Close')
        axes.plot(ticker_data['Start Date'], ticker_data['50_MA'], label=f'{ticker} 50-Day MA')
        axes.plot(ticker_data['Start Date'], ticker_data['200_MA'], label=f'{ticker} 200-Day MA')

    axes.set_title('Adj Close and Moving Averages', fontsize=12)
    axes.set_xlabel('From Date')
    axes.set_ylabel('Price')
    axes.legend(title='Ticker', title_fontsize='10', fontsize='8', loc='upper left')
    axes.grid(True)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45)

def plot_volume_traded(axes, stock_data):
    unique_tickers = stock_data['Ticker'].unique()

    for ticker in unique_tickers:
        ticker_data = stock_data[stock_data['Ticker'] == ticker].copy()
    
    axes.bar(ticker_data.index, ticker_data['Volume'], label='Volume', color='orange')
    axes.set_title(f'{ticker} - Volume Traded')
    axes.set_xlabel('Date')
    axes.set_ylabel('Volume')
    axes.legend()
    axes.grid(True)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45)

def plot_correlation_map(axes, stock_data):
    daily_returns = stock_data.pivot_table(index='Date', columns='Ticker', values='Daily Return')
    correlation_matrix = daily_returns.corr()
    sns.heatmap(correlation_matrix, ax=axes, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f', annot_kws={"size": 10})
    axes.set_title('Correlation Matrix of Daily Returns', fontsize=16)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)

def calc_portfolio_stats(stock_data):
    daily_returns = stock_data.pivot_table(index='Date', columns='Ticker', values='Daily Return')
    expected_returns = daily_returns.mean() * 252  # annualize the returns
    volatility = daily_returns.std() * np.sqrt(252)  # annualize the volatility

    stock_stats = pd.DataFrame({
        'Expected Return': expected_returns,
        'Volatility': volatility
    })

    return stock_stats

# function to calculate portfolio performance
def portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def calc_portfolio_efficient_frontier(initial_allocation, portfolio_weights_df, stock_data,stock_stats,start_date,end_date, previous_start_date, previous_end_date):

    daily_returns = stock_data.pivot_table(index='Date', columns='Ticker', values='Daily Return')
    # number of portfolios to simulate
    num_portfolios = 10000

    # arrays to store the results
    efficient_frontier_results = np.zeros((3, num_portfolios))

    # annualized covariance matrix
    cov_matrix = daily_returns.cov() * 252

    np.random.seed(42)

    for i in range(num_portfolios):

        unique_tickers = stock_data['Ticker'].unique()
        weights = np.random.random(len(unique_tickers))
        weights /= np.sum(weights)

        expected_returns = stock_stats.loc[unique_tickers, 'Expected Return']
        portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)

        
        efficient_frontier_results[0,i] = portfolio_return
        efficient_frontier_results[1,i] = portfolio_volatility
        if portfolio_volatility != 0:
            efficient_frontier_results[2,i] = portfolio_return / portfolio_volatility  # Sharpe Ratio
        else:
            efficient_frontier_results[2,i] = 0  # Handle the case where volatility is zero

    max_sharpe_idx = np.argmax(efficient_frontier_results[2])
    max_sharpe_return = efficient_frontier_results[0, max_sharpe_idx]
    max_sharpe_volatility = efficient_frontier_results[1, max_sharpe_idx]
    max_sharpe_ratio = efficient_frontier_results[2, max_sharpe_idx]
    print(f"Characteristics of portfolio with max Sharpe Ratio: max_sharpe_return={np.round(max_sharpe_return,2)} \
          max_sharpe_volatility={np.round(max_sharpe_volatility,2)} max_sharpe_ratio={np.round(max_sharpe_ratio,2)}")

    max_sharpe_weights = np.zeros(len(unique_tickers))

    for i in range(num_portfolios):
        weights = np.random.random(len(unique_tickers))
        weights /= np.sum(weights)

        portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)

        if efficient_frontier_results[2, i] == max_sharpe_ratio:
            max_sharpe_weights = weights
            break

    df = pd.DataFrame({
        'Ticker': unique_tickers,
        'Weight': np.round(max_sharpe_weights, 4),
        'From Date': start_date,
        'To Date': end_date,
        'Latest Price': 0,
        'Quantity': 0,
        'Value': 0,
    })

    #get the value of each of the tickers in the portfolio at the end_date. The stock data is in the stock_data dataframe
    #get the last row of the stock_data dataframe
    if initial_allocation == True:
        #initial portfolio allocation
        unique_tickers = stock_data['Ticker'].unique()            
        for ticker in unique_tickers:
            stock_data_ticker = stock_data[stock_data['Ticker'] == ticker]
            ticker_value = stock_data_ticker['Adj Close'].iloc[-1]
            #get the weight of the ticker in the portfolio
            ticker_weight = df[df['Ticker'] == ticker]['Weight'].values[0]
            #calculate the quantity of the ticker to buy with an investment of 100,000
            ticker_quantity = 100000 * ticker_weight / ticker_value
            print(f"{ticker} quantity: {ticker_quantity}")
            #calculate the value of the ticker in the portfolio with an investment of 100,000
            ticker_portfolio_value = ticker_value * ticker_quantity
            print(f"{ticker} value: {ticker_portfolio_value}")
            #Update Value to ticker_portfolio_value, where Ticker = ticker
            df.loc[df['Ticker'] == ticker, 'Value'] = np.round(ticker_portfolio_value)
            #Update Quantity to ticker_quantity, where Ticker = ticker
            df.loc[df['Ticker'] == ticker, 'Quantity'] = np.round(ticker_quantity,2)
            #Capture the latest price of the ticker, which is the adjusted close price at the end_date
            df.loc[df['Ticker'] == ticker, 'Latest Price'] = np.round(ticker_value,2)
            #Add two columns called Sold and Bought to the df dataframe and set them to 0
            #Set Bought to 0, where Ticker = ticker
            #Set Sold to 0, where Ticker = ticker
            df.loc[df['Ticker'] == ticker, 'Bought'] = 0
            df.loc[df['Ticker'] == ticker, 'Sold'] = 0
    else:
        #rebalance the portfolio
        #get the unique tickers in the stock_data dataframe
        unique_tickers = stock_data['Ticker'].unique()            
        for ticker in unique_tickers:
            stock_data_ticker = stock_data[stock_data['Ticker'] == ticker]
            ticker_value = stock_data_ticker['Adj Close'].iloc[-1]
            #get the weight of the ticker in the portfolio
            ticker_weight = df[df['Ticker'] == ticker]['Weight'].values[0]
            #calculate the quantity of the ticker to buy or sell to rebalance the portfolio
            #Use the quantity of the ticker for the previous period and compare with the current quantity to
            #decide whether to buy or sell
            #get the ticker values corresponding to the previous period by subtracting the increment from the start_date
            # previous_start_date = start_date - timedelta(days=1)
            # previous_end_date = end_date - timedelta(days=1)
            print(f"Previous start date: {previous_start_date}")
            print(f"Previous end date: {previous_end_date}")
            previous_stock_data = portfolio_weights_df[
                (portfolio_weights_df['From Date'] >= previous_start_date) &
                (portfolio_weights_df['To Date'] <= previous_end_date)
            ]
            
            previous_quantity = previous_stock_data[previous_stock_data['Ticker'] == ticker]['Quantity'].values[0]
            previous_weight = previous_stock_data[previous_stock_data['Ticker'] == ticker]['Weight'].values[0]     
            #compute current quantity using previous quantity and weight
            ticker_quantity = previous_quantity * ticker_weight / previous_weight
            
            #Update Quantity to ticker_quantity, where Ticker = ticker
            df.loc[df['Ticker'] == ticker, 'Quantity'] = np.round(ticker_quantity,2)
            ticker_portfolio_value = ticker_value * ticker_quantity
            #Capture the latest price of the ticker, which is the adjusted close price at the end_date
            df.loc[df['Ticker'] == ticker, 'Latest Price'] = np.round(ticker_value,2)

            #Update Value to ticker_portfolio_value, where Ticker = ticker
            df.loc[df['Ticker'] == ticker, 'Value'] = np.round(ticker_portfolio_value,2)

            #set the quantity to buy or sell to the difference between the current quantity and the previous quantity
            delta_quantity = ticker_quantity - previous_quantity
            print(f"{ticker} delta quantity: {delta_quantity}")
            if delta_quantity > 0:
                print(f"Buy {delta_quantity} of {ticker}")
                #set the Bought column to the quantity
                df.loc[df['Ticker'] == ticker, 'Bought'] = np.round(delta_quantity,2)
                df.loc[df['Ticker'] == ticker, 'Sold'] = 0
            elif delta_quantity < 0:
                print(f"Sell {-delta_quantity} of {ticker}")
                #set the Sold column to the quantity
                df.loc[df['Ticker'] == ticker, 'Sold'] = -np.round(delta_quantity,2)
                df.loc[df['Ticker'] == ticker, 'Bought'] = 0

    portfolio_weights_df = pd.concat([portfolio_weights_df, df])
    return portfolio_weights_df, efficient_frontier_results

def plot_portfolio_efficient_frontier(axes, efficient_frontier_results):
    axes.scatter(efficient_frontier_results[1,:], efficient_frontier_results[0,:], c=efficient_frontier_results[2,:], cmap='YlGnBu', marker='o')
    axes.set_title('Efficient Frontier')
    axes.set_xlabel('Volatility (Standard Deviation)')
    axes.set_ylabel('Expected Return')
    axes.colorbar(axes.collections[0], ax=axes, label='Sharpe Ratio')
    axes.grid(True)

def calc_transaction_costs(transaction_cost_percent, portfolio_weights_df):
    
    # Calculate the transaction costs for each rebalancing period
    # Each row contains the transaction details for a ticker in either Bought or Sold column. 
    # Calculate the transaction cost for each row and add it to a new column called 'Transaction Cost'
    # Calculate the transaction cost as the product of the transaction cost rate and the value of the transaction
    # Add a new column called 'Transaction Cost' to the portfolio_weights_df dataframe
    # Set the value of the 'Transaction Cost' column to the product of the 'Value' and 'Bought' columns where 'Bought' is greater than 0

    #get unique tickers in the portfolio_weights_df dataframe
    unique_tickers = portfolio_weights_df['Ticker'].unique()

    # For each ticker, calculate the transaction cost as the product of the quantity brought or sold and the current value of the ticker and the transaction cost rate
    for ticker in unique_tickers:
        #get unique dates in the 'From Date' column of the portfolio_weights_df dataframe
        unique_dates = portfolio_weights_df['From Date'].unique()
        #for each unique date in the 'From Date' column of the portfolio_weights_df dataframe
        for date in unique_dates:
            #get the rows in the portfolio_weights_df dataframe with the date
            date_data = portfolio_weights_df[portfolio_weights_df['From Date'] == date]
            #get the rows in the date_data dataframe with the ticker
            ticker_data = date_data[date_data['Ticker'] == ticker]
            #get the value of the ticker
            ticker_price = ticker_data['Latest Price'].values[0]
            #get the quantity bought
            bought = ticker_data['Bought'].values[0]
            #get the quantity sold
            sold = ticker_data['Sold'].values[0]
            #calculate the transaction cost for the ticker
            transaction_cost = 0
            if bought > 0:
                transaction_cost = np.round(ticker_price * bought * transaction_cost_percent,2)
            elif sold > 0:
                transaction_cost = np.round(ticker_price * sold * transaction_cost_percent,2)
            #set the transaction cost for the ticker in the portfolio_weights_df dataframe
            portfolio_weights_df.loc[(portfolio_weights_df['From Date'] == date) & (portfolio_weights_df['Ticker'] == ticker), 'Transaction Cost'] = transaction_cost

    return portfolio_weights_df

def plot_results(axes, portfolio_weights_df, portfolio_value_df,portfolio_currency):

    # Create a new subplot for the portfolio values at milestone dates
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 16))
    # sns.set(style='whitegrid')

    # Annotate the ticker on the lineplot
    for ticker in portfolio_weights_df['Ticker'].unique():
        last_occurrence = portfolio_weights_df[portfolio_weights_df['Ticker'] == ticker].iloc[-1]
        axes.annotate(ticker, (last_occurrence['From Date'], last_occurrence['Weight']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    # Plot the portfolio weights as a dashed line graph with a line for each ticker
    sns.lineplot(data=portfolio_weights_df, x='From Date', y='Weight', hue='Ticker', marker='o', ax=axes, linestyle='--')
    axes.set_title('Portfolio Weights and Value Over Time', fontsize=12)
    axes.set_xlabel('From Date', fontsize=12)
    axes.set_ylabel('Weight (%)', fontsize=12)
    axes.grid(True)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45)

    # Create a secondary y-axis for the total portfolio value
    ax2 = axes.twinx()

    # Plot the total value of the portfolio over time on the secondary y-axis
    sns.lineplot(data=portfolio_value_df, x='From Date', y='Total Value', marker='o', ax=ax2, color='black', label='Total Value')
    ax2.set_ylabel(f'Total Value ({portfolio_currency})', fontsize=14)
    ax2.grid(False)
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: "{:,}".format(int(x))))

    #Plot the adjusted total value of the portfolio over time on the secondary y-axis
    sns.lineplot(data=portfolio_value_df, x='From Date', y='Adjusted Total Value', marker='o', ax=ax2, color='red', label='Adjusted Total Value')
    ax2.set_ylabel(f'Total Value ({portfolio_currency})', fontsize=12)
    ax2.grid(False)
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: "{:,}".format(int(x))))

    # # Annotate the total value at each milestone
    # for i, row in portfolio_value_df.iterrows():
    #     ax2.annotate(f"{int(row['Total Value']):,}", (row['From Date'], row['Total Value']), textcoords="offset points", xytext=(0,10), ha='center')

    # Combine legends
    lines, labels = axes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes.legend(lines + lines2, labels + labels2, title='Ticker / Total Value', title_fontsize='10', fontsize='8', loc='upper left')
    ax2.legend().remove()

def get_ticker_data(tickers, timeframes_df, source_files_folder):
    #iterate through the timeframes_df dataframe
    for index, row in timeframes_df.iterrows():
        start_date = row['From Date']
        end_date = row['To Date']

        print(start_date)
        print(end_date)
 
        for ticker in tickers:
            print(ticker)
            data = yf.download(ticker, start=start_date, end=end_date, progress=False,auto_adjust=False)
            filename = f"{source_files_folder}/portfolio_data_{ticker}_{start_date}_{end_date}.csv"
            data.to_csv(filename)
            print("Data saved to: ", filename)

#Combine the data from the different ticker-specific files into a single dataframe for the given time periods
def load_and_transform_data(tickers, timeframes_df, source_files_folder):
    
    #iterate through the timeframes_df dataframe
    for index, row in timeframes_df.iterrows():
        start_date = row['From Date']
        end_date = row['To Date']

        print(start_date)
        print(end_date)
        stock_data = pd.DataFrame()

        for ticker in tickers:
            filename = f'{source_files_folder}/portfolio_data_{ticker}_{start_date}_{end_date}.csv'
            #read each file and concatenate to the data dataframe
            df = pd.read_csv(filename)
            #First row is header row with column names
            # Skip the next 2 rows
            df = df.iloc[2:]
            # Add a column for the ticker and set it to the ticker value
            df['Ticker'] = ticker
            #Change name of 'Price' column to 'Date'
            df.rename(columns={'Price':'Date'}, inplace=True)
            #change datatype of 'Date' column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            df['Adj Close'] = pd.to_numeric(df['Adj Close'])
            df['Close'] = pd.to_numeric(df['Close'])
            df['High'] = pd.to_numeric(df['High'])
            df['Low'] = pd.to_numeric(df['Low'])
            df['Open'] = pd.to_numeric(df['Open'])
            # Concatenate to the main DataFrame
            stock_data = pd.concat([stock_data, df])

        #sort the data by Date and Ticker
        stock_data.sort_values(by=['Date', 'Ticker'], inplace=True)
        #write the stock_data to a csv file
        stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()
        stock_data.to_csv(f'{source_files_folder}/portfolio_data_all_{start_date}_{end_date}.csv')

        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.reset_index(inplace=True)
        stock_data.set_index('Date', inplace=True)

        #drop the column called 'Index'
        stock_data.drop(columns='index', inplace=True)
        #Arrange the remaining columns in the order: Date,Ticker,Adj Close,Close,High,Low,Open,Volume
        stock_data = stock_data[['Ticker', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]

        print(stock_data.columns)
        print(stock_data.info())
        print(stock_data.head())
        print(stock_data.tail())
        #print unique values in Ticker column
        print("Unique values in Ticker column:", stock_data['Ticker'].unique())

def get_timeframes_df(start_date, end_date, increment):
    #calculate number of iterations to run using start_date and today's date
    
    #calculate the number of days between start_date and today's date
    num_days = (end_date - start_date).days
    #calculate the number of increments to run
    num_increments = num_days // increment.days
    #check if there is a nonzero remainder when dividing num_days by increment.days
    remaining_days = num_days % increment.days
    if remaining_days > 0:
        num_increments += 1

    timeframes_list = []
    for i in range(1, num_increments):
        timeframes_list.append({'From Date': start_date, 'To Date': start_date + increment})
        start_date += (increment+timedelta(days=1))

    max_to_date = end_date
    #if number of entries in the timeframes_list is greater than 1
    if len(timeframes_list) > 1:
        #get max 'To Date' in the timeframes_list
        max_to_date = max(timeframes_list, key=lambda x:x['To Date'])['To Date']

    #add any remainder days to the timeframes_list
    #if remaining_days is greater than 0 and max 'To Date' is less than today's date
    #or if remaining_days is greater than 0 and len(timeframes_list) is 0

    if remaining_days > 0 and (max_to_date < end_date or len(timeframes_list) == 0):
        #if From Date is same as To Date, skip the iteration
        if start_date != end_date:
            timeframes_list.append({'From Date': start_date, 'To Date': end_date})

    #Check if any of the intervals in the timeframes_list is less than increment, and delete it
    #get the timeframes with less than increment days
    timeframes_to_be_removed = [timeframe for timeframe in timeframes_list if (timeframe['To Date'] - timeframe['From Date']).days < increment.days]
    print("Removed following timeframes with duration less than increment days:")
    print(timeframes_to_be_removed)
    #remove the timeframes with less than increment days
    timeframes_list = [timeframe for timeframe in timeframes_list if (timeframe['To Date'] - timeframe['From Date']).days >= increment.days]
    timeframes_df = pd.DataFrame(timeframes_list)

    return timeframes_df

def merge_data(timeframes_df, source_files_folder):

    merged_stock_data = pd.DataFrame()
    for index, row in timeframes_df.iterrows():
        start_date = row['From Date']
        end_date = row['To Date']
        df = pd.read_csv(f'{source_files_folder}/portfolio_data_all_{start_date}_{end_date}.csv')
        df['Start Date'] = start_date  # Add start_date to each row
        merged_stock_data = pd.concat([merged_stock_data, df])

    #check for NaN values in the merged_stock_data dataframe and drop them
    print("NaN values in the merged_stock_data dataframe:")
    print(merged_stock_data.isnull().sum())
    merged_stock_data.dropna(inplace=True)

    #remove column named 'Unnamed: 0'
    if 'Unnamed: 0' in merged_stock_data.columns:
        merged_stock_data.drop(columns='Unnamed: 0', inplace=True)

    #convert numeric columns to have only 2 decimal places
    numeric_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    merged_stock_data[numeric_cols] = merged_stock_data[numeric_cols].apply(lambda x: np.round(x,2))
    
    return merged_stock_data

def get_final_results(portfolio_weights_df):

    transaction_cost_percent = float(os.environ.get('transaction_cost_percent'))
    output_files_folder = os.environ.get('output_files_folder')
     # Sort the portfolio_weights_df by 'From Date' and 'Ticker'
    portfolio_weights_df.sort_values(by=['From Date', 'Ticker'], inplace=True)
    # Reset the index of the portfolio_weights_df
    portfolio_weights_df.reset_index(drop=True, inplace=True)
    print(portfolio_weights_df)

    #get the transaction costs for the portfolio
    # Assume a transaction cost of 1% of the value of the transaction
    portfolio_weights_df = calc_transaction_costs(transaction_cost_percent, portfolio_weights_df)    

    # Create a new dataframe called portfolio_value_df to be used to store the total value of the portfolio at each milestone date
    portfolio_value_list = []
    # Get the unique dates in the 'From Date' column of the portfolio_weights_df dataframe
    unique_dates = portfolio_weights_df['From Date'].unique()
    # Loop through the unique dates
    for date in unique_dates:
        # Get the rows in the portfolio_weights_df dataframe with the date
        portfolio_weights = portfolio_weights_df[portfolio_weights_df['From Date'] == date]
        # Calculate the total value of the portfolio on that date
        total_value = portfolio_weights['Value'].sum()
        total_transaction_cost = portfolio_weights['Transaction Cost'].sum()
        #get the unique tickers in the portfolio_weights dataframe
        unique_tickers = portfolio_weights['Ticker'].unique()
        adjusted_total_value = (total_value - total_transaction_cost)
        # Add a row to the portfolio_value_list with the date 
        portfolio_value_list.append({'From Date': date, 'Total Value': np.round(total_value,2), 'Total Transaction Cost': np.round(total_transaction_cost,2), 'Adjusted Total Value': np.round(adjusted_total_value,2)})
    
    portfolio_value_df = pd.DataFrame(portfolio_value_list)

    #Save the portfolio_weights_df and portfolio_value_df to csv files
    portfolio_weights_df.to_csv(f'{output_files_folder}/portfolio_weights_results.csv')
    portfolio_value_df.to_csv(f'{output_files_folder}/portfolio_value_results.csv')
    print(portfolio_value_df)

    return portfolio_weights_df, portfolio_value_df

#create a class to hold portfolio metadata
class Portfolio:
    portfolio_value = 0
    portfolio_allocation = {}

    def __init__(self, name, currency, tickers):
        self.name = name
        self.currency = currency
        self.tickers = tickers

    #function to set initial portfolio allocation
    #the function takes a dictionary of tickers and positions as input
    #positions is a tuple of the form (quantity, price)
    def set_initial_allocation(self, allocation):
        self.portfolio_allocation = allocation
        self.calculate_value()
        self.calculate_ticker_weights()

    #function to calculate current value of the portfolio
    #it is obtained by multiplying the quantity of each ticker by the current price of the ticker
    #the price and quantity are obtained from the allocation dictionary
    def calculate_value(self):
        for ticker in self.tickers:
            quantity, price = self.portfolio_allocation[ticker]
            #get the current price of the ticker
            #get the current quantity of the ticker
            #calculate the value of the ticker
            #add the value of the ticker to the total value of the portfolio
            ticker_value = price * quantity
            self.portfolio_value += np.round(ticker_value,2)

    def calculate_ticker_weights(self):
        #Add weights to the initial allocation dictionary
        for ticker in self.tickers:
            quantity, price = self.portfolio_allocation[ticker]
            ticker_value = price * quantity
            ticker_weight = np.round((ticker_value / self.portfolio_value)*100,2)
            self.portfolio_allocation[ticker] = (quantity, price, ticker_value, ticker_weight)

    def get_portfolio_weights(self):
        #weights are stored in the allocation dictionary
        ticker_weights ={}
        for ticker in self.tickers:
            quantity, price, value, weight = self.portfolio_allocation[ticker]
            ticker_weights[ticker] = weight
        return ticker_weights

def prepare_data(load_from_files):

    #load the configuration parameters
    source_files_folder = os.environ.get('source_files_folder') #'source_files'
    output_files_folder = os.environ.get('output_files_folder') #source_files'
    start_date = date.fromisoformat( os.environ.get('start_date'))
    increment = timedelta(days=int(os.environ.get('increment'))) #timedelta(days=30)
    end_date = date.fromisoformat(os.environ.get('end_date'))
    timeframes_df = get_timeframes_df(start_date, end_date, increment)
    tickers = os.environ.get('tickers').split(',') #['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']

    if load_from_files == False:
        #get each ticker data for the given time period, in separate files
        get_ticker_data(tickers, timeframes_df, source_files_folder)
        
    #pivot data so that all the tickers are in the same dataframe and save to a single file per time period
    load_and_transform_data(tickers, timeframes_df, source_files_folder)
    merged_stock_data = merge_data(timeframes_df, source_files_folder)

    #save merged_stock_data to a csv file
    merged_stock_data.to_csv(f'{output_files_folder}/merged_stock_data.csv')

    return merged_stock_data

def run_portfolio_analysis(load_data):

    if load_data == True:
        merged_stock_data = prepare_data(load_data)
        
    #load the configuration parameters
    source_files_folder = os.environ.get('source_files_folder') #'source_files'
    output_files_folder = os.environ.get('output_files_folder') #source_files'
    currency = os.environ.get('currency')
    tickers = os.environ.get('tickers').split(',') 
    initial_allocation = json.loads(os.environ.get('initial_allocation'))
    transaction_cost_percent = float(os.environ.get('transaction_cost_percent'))
    start_date = date.fromisoformat(os.environ.get('start_date')) 
    end_date = date.fromisoformat(os.environ.get('end_date'))
    increment = timedelta(days=int(os.environ.get('increment')))
    timeframes_df = get_timeframes_df(start_date, end_date, increment)

    portfolio = Portfolio('Portfolio#1', currency, tickers)
    #setting the initial allocation also automatically calculates the portfolio value and weights
    portfolio.set_initial_allocation(initial_allocation) 

    #update portfolio allocation with weights
    print(f"Initial portfolio value: {portfolio.currency} {portfolio.portfolio_value}")

    #get the portfolio weights
    portfolio_weights = portfolio.get_portfolio_weights()
    print("Portfolio weights:")
    print(portfolio_weights)

    portfolio_weights_df = pd.DataFrame()
    initial_allocation = True
    #iterate through the timeframes_df dataframe
    for index, row in timeframes_df.iterrows():
        start_date = row['From Date']
        end_date = row['To Date']
        filename = f'{source_files_folder}/portfolio_data_all_{start_date}_{end_date}.csv'
        #--> filename = "./stock_data.csv" #file for backtesting
        stock_data = pd.read_csv(filename)
        #--> uncomment calc for backtesting
        #--> stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()

        #get previous start date and end date as prior rows in the timeframes_df dataframe
        if initial_allocation == True:
            # Handle the case where there is no previous row
            previous_start_date = None
            previous_end_date = None
        else:
            # Get previous start date and end date as prior rows in the timeframes_df dataframe
            previous_start_date = timeframes_df.loc[index-1, 'From Date']
            previous_end_date = timeframes_df.loc[index-1, 'To Date']

        print("Calculating portfolio efficient frontier for the period: ", start_date, end_date)
        print("Previous dates are : ", previous_start_date, previous_end_date)
        
        stock_stats = calc_portfolio_stats(stock_data)
        #save the stock_stats to a csv file
        stock_stats.to_csv(f'{output_files_folder}/stock_stats_{start_date}_{end_date}.csv')
        print(stock_stats)

        portfolio_weights_df, efficient_frontier_results = calc_portfolio_efficient_frontier(initial_allocation, portfolio_weights_df, stock_data, stock_stats, start_date, end_date,  previous_start_date, previous_end_date)        
        # -- plot_portfolio_efficient_frontier(axes[2, 1], efficient_frontier_results)
        print("Best portfolio:")
        #Add a column called % to the portfolio_weights_df dataframe
        portfolio_weights_df['%'] = np.round(portfolio_weights_df['Weight'] * 100,2)
        # plt.tight_layout()
        # plt.show()
        initial_allocation = False

    portfolio_weights_df, portfolio_value_df = get_final_results(portfolio_weights_df)

    portfolio_weights_delta = get_portfolio_weights_delta(portfolio_weights_df)
    return portfolio_weights_df, portfolio_value_df, portfolio_weights_delta

def plot_graphs(currency):

     # Create a figure with a custom layout capable of fisplaying multiple graphs
    fig = plt.figure(figsize=(14, 20))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 2, 2, 2], width_ratios=[1])
    sns.set(style='whitegrid')
    
    output_files_folder = os.environ.get('output_files_folder')
    
    merged_stock_data = pd.read_csv(f'{output_files_folder}/merged_stock_data.csv')
    #plot individual, ticker-level plots
    ax1 = plt.subplot(gs[0, :]) #first row, all columns
    plot_time_series(ax1, merged_stock_data, currency)
    ax2 = plt.subplot(gs[1, :]) #second row, all columns
    plot_daily_returns(ax2, merged_stock_data)
    # ax3 = plt.subplot(gs[2, :]) #third row, all columns
    #TODO: Fix the plot_moving_averages function
    # short_window = 50
    # long_window = 200
    # plot_moving_averages(ax3, merged_stock_data, short_window, long_window)
    #TODO: Fix the plot_volume_traded function
    # -- plot_volume_traded(axes[1, 0], stock_data)
    
    # -- plot_correlation_map(axes[2, 0], stock_data)

    ax4 = fig.add_subplot(gs[2, :]) #row 3, all columns
    plot_results(ax4, portfolio_weights_df, portfolio_value_df, currency)

    # Adjust layout to ensure all subplots are visible
    plt.tight_layout()

    # Save the plots to a PDF file
    pdf_filename = f"{output_files_folder}/portfolio_plots.pdf"
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)

    plt.show()
    
    print(f"Plots saved to {pdf_filename}")

def load_configuration_params():

    #get all configuration parameters from the config.properties file as strings as they will be set as environment variables
    config_params = {}

    config = configparser.ConfigParser()
    config.read('/Users/shriniwasiyengar/git/python_ML/fin-ml/portfolio-analyzer/backend/config.properties')

      # define the time period for the data
    start_date = config.get('Portfolio', 'start_date') #'2020-02-16'
    end_date = str(date.today())
    #Backtesting dates: 3rd July 2023 to 29th June 2024
    #uncomment for backtesting
    #--> start_date = date.fromisoformat('2023-07-03') #backtesting start date
    #--> end_date = date.fromisoformat('2024-06-29') #backtesting end date. Use one date more than the actual end date
    config_params['start_date'] = start_date
    config_params['end_date'] = end_date

    increment = config.get('Portfolio', 'increment') #365
    config_params['increment'] = increment

    transaction_cost_percent = config.get('Portfolio', 'transaction_cost_percent') #0.01 
    config_params['transaction_cost_percent'] = transaction_cost_percent

    # tickers = ['RELIANCE.NS', 'TCS.NS','INFY.NS', 'HDFCBANK.NS']
    # tickers = ['AAPL','BAC', 'BK', 'LIT', 'VTSAX', 'CSCO', 'GIS','SONY','INTC']
    tickers = config.get('Portfolio', 'tickers')#['BAC', 'LIT', 'VTSAX', 'CSCO', 'GIS','SONY','INTC']
    config_params['tickers'] = tickers
    
    currency = config.get('Portfolio', 'currency') #'USD'
    config_params['currency'] = currency

    #Todo: load initial allocation from config file or csv file
    initial_allocation = {
        # 'AAPL': (245.035, 80.689),
        # 'BK': (88.9, 1534.2394),
        'BAC': (45.985, 52.079),
        'LIT': (42.34, 20.336),
        'VTSAX': (147.36, 71.747),
        'CSCO': (64.645, 28.957),
        'GIS': (57.88, 35.751),
        'SONY': (24.74, 56.521),
        'INTC': (26.0101, 10)
    }
    #convert initial_allocation to a string before setting as environment variable
    initial_allocation_str = json.dumps(initial_allocation)
    config_params['initial_allocation'] = initial_allocation_str

    config_params['source_files_folder'] = config.get('Portfolio', 'source_files_folder') #'source_files'
    config_params['output_files_folder'] = config.get('Portfolio', 'output_files_folder') #'output_files'

    print("Configuration parameters:")
    print(config_params)

    #create and set environment variables based on the configuration parameters
    os.environ['start_date'] = config_params['start_date']
    os.environ['end_date'] = config_params['end_date']
    os.environ['increment'] = config_params['increment']
    os.environ['transaction_cost_percent'] = config_params['transaction_cost_percent']
    os.environ['tickers'] = config_params['tickers']
    os.environ['currency'] = config_params['currency']
    os.environ['initial_allocation'] = config_params['initial_allocation']
    os.environ['source_files_folder'] = config_params['source_files_folder']
    os.environ['output_files_folder'] = config_params['output_files_folder']

def get_portfolio_weights_delta(portfolio_weights_df):

    start_date = date.fromisoformat(os.environ.get('start_date'))
    end_date = date.fromisoformat(os.environ.get('end_date'))
    increment = timedelta(days=int(os.environ.get('increment')))
    timeframes_df = get_timeframes_df(start_date, end_date, increment)
    tickers = os.environ.get('tickers').split(',')
    #get the last period in the timeframes_df dataframe
    final_period = timeframes_df.iloc[-1]
    #get the start date and end date of the final period
    final_start_date = final_period['From Date']
    final_end_date = final_period['To Date']
    
    #get initial weights from the initial allocation dictionary in the Portfolio class
    #create a table to display ticker, initial weights and final weights
    initial_portfolio_weights_df = portfolio_weights_df[portfolio_weights_df['From Date'] == start_date]
    final_portfolio_weights_df = portfolio_weights_df[portfolio_weights_df['From Date'] == final_start_date]
    table = pd.DataFrame()
    table['Ticker'] = tickers
    table['Initial Weight'] = [initial_portfolio_weights_df[initial_portfolio_weights_df['Ticker'] == ticker]['Weight'].values[0] for ticker in tickers]
    table['Final Weight'] = [final_portfolio_weights_df[final_portfolio_weights_df['Ticker'] == ticker]['Weight'].values[0] for ticker in tickers]
    print(table)

    return table

#add main function to run the code
if __name__ == "__main__":

    load_configuration_params()
    load_from_files = True
    prepare_data(load_from_files)
    #convert tickers string to list before using in the run_portfolio_analysis function
    load_data = False
    portfolio_weights_df, portfolio_value_df, portfolio_weights_delta = run_portfolio_analysis(load_data)    
    
    plot_graphs(os.environ.get('currency'))
