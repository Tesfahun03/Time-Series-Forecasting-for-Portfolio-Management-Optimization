import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

logging.basicConfig(
    filename='../logs/eda.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info(
    '****************************Logging started for EDA module****************************')


class EDA:
    """
    Class for exploratory data analysis on financial time series.

    Attributes:
        data (pd.DataFrame): Input data for analysis.
    """

    def __init__(self, data):
        self.data = data
        logging.info("EDA instance created.")

    def plot_closing_price(self, ticker):
        """Visualizes closing price trends over time."""
        logging.info("Plotting closing price for %s", ticker)
        try:
            ticker_data = self.data[self.data['Ticker'] == ticker]
            plt.figure(figsize=(10, 6))
            plt.plot(ticker_data['Date'],
                     ticker_data['Close'], label=f'{ticker} Close')
            plt.title(f'{ticker} Closing Price Over Time')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            plt.savefig(f'../images/{ticker}_close_price.png')
            plt.close()
            logging.info("Closing price plot saved for %s", ticker)
        except Exception as e:
            logging.error("Error plotting closing price: %s", e)
            raise

    def calculate_daily_returns(self):
        """Calculates and plots daily percentage changes."""
        logging.info("Calculating daily returns.")
        try:
            self.data['Daily_Return'] = self.data.groupby(
                'Ticker')['Close'].pct_change()
            for ticker in self.data['Ticker'].unique():
                ticker_data = self.data[self.data['Ticker'] == ticker]
                plt.figure(figsize=(10, 6))
                plt.plot(
                    ticker_data['Date'], ticker_data['Daily_Return'], label=f'{ticker} Returns')
                plt.title(f'{ticker} Daily Returns')
                plt.xlabel('Date')
                plt.ylabel('Daily Return')
                plt.legend()
                plt.savefig(f'../images/{ticker}_daily_returns.png')
                plt.close()
            logging.info("Daily returns calculated and plotted.")
            return self.data
        except Exception as e:
            logging.error("Error calculating daily returns: %s", e)
            raise

    def analyze_volatility(self, window=20):
        """Calculates rolling mean and standard deviation for volatility."""
        logging.info("Analyzing volatility with window=%d", window)
        try:
            self.data['Rolling_Mean'] = self.data.groupby('Ticker')['Daily_Return'].rolling(
                window).mean().reset_index(level=0, drop=True)
            self.data['Rolling_Std'] = self.data.groupby('Ticker')['Daily_Return'].rolling(
                window).std().reset_index(level=0, drop=True)
            logging.info("Volatility analysis completed.")
            return self.data
        except Exception as e:
            logging.error("Error analyzing volatility: %s", e)
            raise

    def detect_outliers(self, threshold=3):
        """Detects outliers in daily returns using Z-score."""
        logging.info("Detecting outliers with threshold=%d", threshold)
        try:
            self.data['Z_Score'] = np.abs(
                (self.data['Daily_Return'] - self.data['Rolling_Mean']) / self.data['Rolling_Std'])
            outliers = self.data[self.data['Z_Score'] > threshold]
            logging.info("Found %d outliers.", len(outliers))
            return outliers
        except Exception as e:
            logging.error("Error detecting outliers: %s", e)
            raise

    def decompose_series(self, ticker, period=252):
        """Decomposes time series into trend, seasonal, and residual components."""
        logging.info("Decomposing series for %s with period=%d",
                     ticker, period)
        try:
            ticker_data = self.data[self.data['Ticker']
                                    == ticker]['Close'].dropna()
            decomposition = seasonal_decompose(
                ticker_data, model='additive', period=period)
            decomposition.plot()
            plt.savefig(f'../images/{ticker}_decomposition.png')
            plt.close()
            logging.info("Time series decomposition completed for %s", ticker)
            return decomposition
        except Exception as e:
            logging.error("Error decomposing series: %s", e)
            raise

    def calculate_metrics(self, risk_free_rate=0.02):
        """Calculates VaR and Sharpe Ratio."""
        logging.info("Calculating VaR and Sharpe Ratio.")
        try:
            metrics = {}
            for ticker in self.data['Ticker'].unique():
                ticker_data = self.data[self.data['Ticker']
                                        == ticker]['Daily_Return'].dropna()
                # VaR (95% confidence)
                var = np.percentile(ticker_data, 5)
                # Annualized Sharpe Ratio
                excess_return = ticker_data.mean() * 252 - risk_free_rate
                volatility = ticker_data.std() * np.sqrt(252)
                sharpe = excess_return / volatility
                metrics[ticker] = {'VaR (95%)': var, 'Sharpe Ratio': sharpe}
            logging.info("Metrics calculated: %s", metrics)
            return metrics
        except Exception as e:
            logging.error("Error calculating metrics: %s", e)
            raise
