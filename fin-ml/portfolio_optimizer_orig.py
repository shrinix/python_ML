from datetime import date, timedelta
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def get_ticker_data(tickers, start_date, end_date):

    for ticker in tickers:
        print(ticker)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False,auto_adjust=False)
        filename = f"{source_folder}/portfolio_data"+ticker+".csv"
        data.to_csv(filename)
        print("Data saved to: ", filename)

    stock_data = pd.DataFrame()
    for ticker in tickers:
        filename = f"{source_folder}/portfolio_data"+ticker+".csv"
        #read each file and concatenate to the data dataframe
        df = pd.read_csv(filename)
        #First row is header row with column names
        # # Drop the next 2 rows
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
    return stock_data

def plot_time_series(axes,stock_data):
    # plt.figure(figsize=(14, 7))
    sns.set_theme(style='whitegrid')

    sns.lineplot(ax=axes, data=stock_data, x='Date', y='Adj Close', hue='Ticker', marker='o')

    axes.set_title('Adj Close Price Over Time', fontsize=16)
    axes.set_xlabel('Date', fontsize=14)
    axes.set_ylabel('Adj Close Price', fontsize=14)
    axes.legend(title='Ticker', title_fontsize='13', fontsize='11')
    axes.grid(True)

    axes.set_xticklabels(axes.get_xticklabels(), rotation=45)

def plot_daily_returns(axes, stock_data):

    unique_tickers = stock_data['Ticker'].unique()
    for ticker in unique_tickers:
        ticker_data = stock_data[stock_data['Ticker'] == ticker]
        sns.histplot(ticker_data['Daily Return'].dropna(), ax=axes, bins=50, kde=True, label=ticker, alpha=0.5)

    axes.set_title('Distribution of Daily Returns', fontsize=16)
    axes.set_xlabel('Daily Return', fontsize=14)
    axes.set_ylabel('Frequency', fontsize=14)
    axes.legend(title='Ticker', title_fontsize='13', fontsize='11')
    axes.grid(True)
    
def plot_moving_averages(axes, stock_data, short_window, long_window):

    unique_tickers = stock_data['Ticker'].unique()

    for ticker in unique_tickers:
        ticker_data = stock_data[stock_data['Ticker'] == ticker].copy()
        ticker_data['50_MA'] = ticker_data['Adj Close'].rolling(window=short_window).mean()
        ticker_data['200_MA'] = ticker_data['Adj Close'].rolling(window=long_window).mean()

        axes.plot(ticker_data.index, ticker_data['Adj Close'], label='Adj Close')
        axes.plot(ticker_data.index, ticker_data['50_MA'], label='50-Day MA')
        axes.plot(ticker_data.index, ticker_data['200_MA'], label='200-Day MA')
        axes.set_title(f'{ticker} - Adj Close and Moving Averages')
        axes.set_xlabel('Date')
        axes.set_ylabel('Price')
        axes.legend()
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

def calc_portfolio_efficient_frontier(initial_allocation, portfolio_weights_df, stock_data,stock_stats,start_date,end_date):

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
        efficient_frontier_results[2,i] = portfolio_return / portfolio_volatility  # Sharpe Ratio

    max_sharpe_idx = np.argmax(efficient_frontier_results[2])
    max_sharpe_return = efficient_frontier_results[0, max_sharpe_idx]
    max_sharpe_volatility = efficient_frontier_results[1, max_sharpe_idx]
    max_sharpe_ratio = efficient_frontier_results[2, max_sharpe_idx]
    print("Characteristics of portfolio with max Sharpe Ratio: ", max_sharpe_return, max_sharpe_volatility, max_sharpe_ratio)

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
            df.loc[df['Ticker'] == ticker, 'Value'] = np.round(ticker_portfolio_value,2)
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
            previous_start_date = start_date - increment - timedelta(days=1)
            previous_end_date = end_date - increment - timedelta(days=1)
            print("Calculating portfolio efficient frontier for the period: ", start_date, end_date)
            print("Previous dates are : ", previous_start_date, previous_end_date)
        
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
    fig.colorbar(axes.collections[0], ax=axes, label='Sharpe Ratio')
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

def plot_results(axes,portfolio_weights_df):

    # Sort the portfolio_weights_df by 'From Date' and 'Ticker'
    portfolio_weights_df.sort_values(by=['From Date', 'Ticker'], inplace=True)
    # Reset the index of the portfolio_weights_df
    portfolio_weights_df.reset_index(drop=True, inplace=True)
    print(portfolio_weights_df)

    #get the transaction costs for the portfolio

    # Assume a transaction cost of 1% of the value of the transaction
    transaction_cost_percent = 0.01
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
        adjusted_total_value = total_value - total_transaction_cost
        # Add a row to the portfolio_value_list with the date 
        portfolio_value_list.append({'From Date': date, 'Total Value': total_value, 'Total Transaction Cost': total_transaction_cost, 'Adjusted Total Value': adjusted_total_value})
    portfolio_value_df = pd.DataFrame(portfolio_value_list)

    #Save the portfolio_weights_df and portfolio_value_df to csv files
    portfolio_weights_df.to_csv(f"{dest_folder}/portfolio_weights_results_orig.csv")
    portfolio_value_df.to_csv(f"{dest_folder}/portfolio_value_results_orig.csv")
    print(portfolio_value_df)

    # Create a new subplot for the portfolio values at milestone dates
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 16))
    # sns.set(style='whitegrid')

    # Plot the portfolio weights as a dashed line graph with a line for each ticker
    sns.lineplot(data=portfolio_weights_df, x='From Date', y='Weight', hue='Ticker', marker='o', ax=axes, linestyle='--')
    axes.set_title('Portfolio Weights and Value Over Time', fontsize=16)
    axes.set_xlabel('From Date', fontsize=14)
    axes.set_ylabel('Weight (%)', fontsize=14)
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
    ax2.set_ylabel(f'Total Value ({portfolio_currency})', fontsize=14)
    ax2.grid(False)
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: "{:,}".format(int(x))))

    # # Annotate the total value at each milestone
    # for i, row in portfolio_value_df.iterrows():
    #     ax2.annotate(f"{int(row['Total Value']):,}", (row['From Date'], row['Total Value']), textcoords="offset points", xytext=(0,10), ha='center')

    # Combine legends
    lines, labels = axes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes.legend(lines + lines2, labels + labels2, title='Ticker / Total Value', title_fontsize='13', fontsize='11', loc='upper left')
    ax2.legend().remove()

#add main function to run the code
if __name__ == "__main__":

    # define the time period for the data
    start_date = date.fromisoformat('2020-02-16')
    #get today's date
    today_date = date.today()
    end_date = today_date
    #Backtesting dates: 3rd July 2023 to 29th June 2024
    #Uncomment for backtesting
    #--> start_date = date.fromisoformat('2023-07-03')
    #--> end_date = date.fromisoformat('2024-06-27')

    increment = timedelta(days=365)
    portfolio_weights_df = pd.DataFrame()
    portfolio_currency = 'INR'

    initial_allocation = True
    load_from_files = True
    #calculate number of iterations to run using start_date and today's date
    
    #calculate the number of days between start_date and today's date
    num_days = (end_date - start_date).days
    #calculate the number of increments to run
    num_increments = num_days // increment.days
    #create a plot capable of fisplaying multiple graphs
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))
    sns.set(style='whitegrid')

    source_folder = "orig"
    dest_folder = "orig"
    for i in range(0, num_increments):

        # start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        # end_date = date.today().strftime("%Y-%m-%d")
        # list of stock tickers to download
        end_date = start_date + increment
        tickers = ['RELIANCE.NS', 'TCS.NS','INFY.NS', 'HDFCBANK.NS']
        print(start_date)
        print(end_date)

        if load_from_files == True:
            stock_data = pd.read_csv(f'{source_folder}/portfolio_data_all_{start_date}_{end_date}.csv')
        else:
            stock_data = get_ticker_data(tickers, start_date, end_date)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.reset_index(inplace=True)
        stock_data.set_index('Date', inplace=True)

        #drop the column called 'Index'
        stock_data.drop(columns='index', inplace=True)
        #Arrange the remaining columns in the order: Date,Ticker,Adj Close,Close,High,Low,Open,Volume
        stock_data = stock_data[['Ticker', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]

        #sort the data by Date and Ticker
        stock_data.sort_values(by=['Date', 'Ticker'], inplace=True)
        #write the stock_data to a csv file
        stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()
        stock_data.to_csv(f'{source_folder}/portfolio_data_all_{start_date}_{end_date}.csv')

        print(stock_data.columns)
        print(stock_data.info())
        print(stock_data.head())
        print(stock_data.tail())

        #print unique values in Ticker column
        print("Unique values in Ticker column:", stock_data['Ticker'].unique())

        short_window = 50
        long_window = 200

        # -- plot_time_series(axes[0,0], stock_data)
        #TODO: Fix the plot_moving_averages function
        # -- plot_moving_averages(axes[0,1], stock_data, short_window, long_window)
        #TODO: Fix the plot_volume_traded function
        # -- plot_volume_traded(axes[1, 0], stock_data)
        # -- plot_daily_returns(axes[1, 1], stock_data)
        # -- plot_correlation_map(axes[2, 0], stock_data)

        stock_stats = calc_portfolio_stats(stock_data)
        print(stock_stats)
        stock_stats.to_csv(f"{dest_folder}/stock_stats_{start_date}_{end_date}.csv")

        if i == 0:
            initial_allocation = True
        else:
            initial_allocation = False
        portfolio_weights_df, efficient_frontier_results = calc_portfolio_efficient_frontier(initial_allocation, portfolio_weights_df, stock_data, stock_stats,start_date,end_date)        
        # -- plot_portfolio_efficient_frontier(axes[2, 1], efficient_frontier_results)
        print("Best portfolio:")
        #Add a column called % to the portfolio_weights_df dataframe
        portfolio_weights_df['%'] = np.round(portfolio_weights_df['Weight'] * 100,2)
        # plt.tight_layout()
        # plt.show()
        
        start_date += increment+timedelta(days=1)

    plot_results(axes,portfolio_weights_df)

    # Adjust layout to ensure all subplots are visible
    plt.tight_layout()

    # Save the plots to a PDF file
    pdf_filename = f"{dest_folder}/portfolio_plots_orig.pdf"
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)

    plt.show()
    
    print(f"Plots saved to {pdf_filename}")
