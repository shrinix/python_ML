import yfinance as yf

class DividendStock:
    def __init__(self, ticker):
        """
        Initialize stock parameters by fetching real-time data.
        :param ticker: Stock symbol (string)
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

        # Fetch stock price, EPS, and DPS
        self.price = self.stock.history(period="1d")["Close"].iloc[-1]
        self.eps = self.stock.info.get("trailingEps", None)
        self.dps = self.get_dividend_per_share()
        self.fcfps = None  # Not directly available via Yahoo Finance

    def get_dividend_per_share(self):
        """Fetches the latest dividend per share from Yahoo Finance."""
        dividends = self.stock.dividends
        if dividends.empty:
            return 0
        return dividends.iloc[-1]

    def dividend_yield(self):
        """Calculate Dividend Yield (%)"""
        return (self.dps / self.price) * 100 if self.price > 0 else None

    def dividend_coverage_ratio(self):
        """Calculate Dividend Coverage Ratio (DCR)"""
        return self.eps / self.dps if self.dps > 0 and self.eps > 0 else None

    def payout_ratio(self):
        """Calculate Payout Ratio (%)"""
        return (self.dps / self.eps) * 100 if self.eps and self.dps > 0 else None

    def historical_dividend_growth(self, years=5):
        """Calculate the average dividend growth rate over a given period."""
        dividends = self.stock.dividends
        if len(dividends) < years:
            return None  # Not enough data
        past_dividends = dividends.resample('Y').sum().tail(years)
        if len(past_dividends) < 2:
            return None

        initial_dividend = past_dividends.iloc[0]
        final_dividend = past_dividends.iloc[-1]

        if initial_dividend == 0:
            return None  # Avoid division by zero

        # CAGR formula: (Ending Value / Beginning Value)^(1/n) - 1
        growth_rate = ((final_dividend / initial_dividend) ** (1 / (years - 1))) - 1
        return growth_rate * 100

    def evaluate_stock(self):
        """Evaluate the stock based on key dividend metrics."""

        evaluation = []
        yield_value = self.dividend_yield()
        dcr_value = self.dividend_coverage_ratio()
        payout_value = self.payout_ratio()
        growth_rate = self.historical_dividend_growth()

        # Dividend Yield Assessment
        if yield_value and yield_value > 7:
            evaluation.append("✔ High Dividend Yield (Potentially Risky)")
        elif yield_value and 3 <= yield_value <= 7:
            evaluation.append("✔ Healthy Dividend Yield")
        else:
            evaluation.append("⚠ Low Dividend Yield")

        # Dividend Coverage Ratio (DCR) Assessment
        if dcr_value and dcr_value > 2:
            evaluation.append("✔ Strong Dividend Coverage (Safe)")
        elif dcr_value and 1.5 <= dcr_value <= 2:
            evaluation.append("✔ Sufficient Dividend Coverage")
        else:
            evaluation.append("⚠ Low Dividend Coverage (Risk of Cut)")

        # Payout Ratio Assessment
        if payout_value and payout_value < 50:
            evaluation.append("✔ Low Payout Ratio (Sustainable)")
        elif payout_value and 50 <= payout_value <= 75:
            evaluation.append("✔ Moderate Payout Ratio")
        else:
            evaluation.append("⚠ High Payout Ratio (Risk of Unsustainability)")

        # Dividend Growth Rate Assessment
        if growth_rate and growth_rate > 5:
            evaluation.append("✔ Strong Dividend Growth (>5% annually)")
        elif growth_rate and 2 <= growth_rate <= 5:
            evaluation.append("✔ Moderate Dividend Growth (2-5%)")
        else:
            evaluation.append("⚠ Weak Dividend Growth (<2%)")

        return evaluation

    def print_stock_data(self):
        """Print stock data for the given ticker."""
        print(f"\n📈 **{self.ticker} Stock Data**:")
        print(f"  - Price: ${self.price:.2f}")
        print(f"  - EPS: ${self.eps:.2f}")
        print(f"  - Dividend Per Share (DPS): ${self.dps:.2f}")
                                                
    def display_metrics_and_evaluation(self):
        """Display calculated dividend metrics and evaluation."""
        print(f"\n📊 **Dividend Metrics for {self.ticker}**:")
        print(f"  - Price: ${self.price:.2f}")
        print(f"  - Dividend Per Share (DPS): ${self.dps:.2f}")
        print(f"  - Dividend Yield: {self.dividend_yield():.2f}%")
        print(f"  - Dividend Coverage Ratio (DCR): {self.dividend_coverage_ratio():.2f}")
        print(f"  - Payout Ratio: {self.payout_ratio():.2f}%")
        growth = self.historical_dividend_growth()
        print(f"  - 5-Year Dividend Growth Rate: {growth:.2f}% (if available)")

        print("\n📈 **Stock Evaluation:**")
        for criterion in self.evaluate_stock():
            print(f"  {criterion}")

# Example usage for SPE and BTO stocks
spe = DividendStock("SPE")
bto = DividendStock("BTO")

#print Stock data
spe.print_stock_data()
bto.print_stock_data()

# Display metrics and evaluation
spe.display_metrics_and_evaluation()
bto.display_metrics_and_evaluation()