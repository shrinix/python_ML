import numpy as np

# Define sample data
weights = np.array([0.4, 0.3, 0.3])  # Portfolio weights for three assets
returns = np.array([0.12, 0.10, 0.15])  # Expected returns for the three assets
cov_matrix = np.array([
    [0.005, -0.010, 0.004],
    [-0.010, 0.040, -0.002],
    [0.004, -0.002, 0.023]
])  # Covariance matrix for the three assets

# Define the portfolio_performance function
def portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Calculate portfolio performance
portfolio_return, portfolio_volatility = portfolio_performance(weights, returns, cov_matrix)

# Print the results
print(f"Portfolio Expected Return: {portfolio_return:.2%}")
print(f"Portfolio Volatility (Risk): {portfolio_volatility:.2%}")