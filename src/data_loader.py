import pandas as pd
import yfinance as yf
import logging

logging.basicConfig(
    filename='../logs/data_loader.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info(
    '****************************Logging started for Data Loader module****************************')


class DataLoader:
    """
    Class to fetch and load financial time series data from YFinance.

    Attributes:
        tickers (list): List of asset tickers (e.g., ['TSLA', 'BND', 'SPY']).
        start_date (str): Start date for data.
        end_date (str): End date for data.
        data (pd.DataFrame): Loaded financial data.
    """

    def __init__(self, tickers, start_date='2015-01-01', end_date='2025-01-31'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        logging.info(
            "DataLoader instance created with tickers: %s", self.tickers)

    def load_data(self):
        """Fetches historical data from YFinance."""
        logging.info("Starting data fetch from YFinance for %s", self.tickers)
        try:
            self.data = yf.download(
                self.tickers, start=self.start_date, end=self.end_date)
            # Simplify multi-index dataframe (if multiple tickers) into a flat structure
            if len(self.tickers) > 1:
                self.data = self.data.stack(level=1).reset_index().rename(
                    columns={'level_1': 'Ticker'})
            logging.info("Data fetched successfully. Shape: %s",
                         self.data.shape)
            self.data.to_csv('../data/financial_data.csv', index=False)
            logging.info("Data saved to ../data/financial_data.csv")
            return self.data
        except Exception as e:
            logging.error("Error fetching data: %s", e)
            raise
