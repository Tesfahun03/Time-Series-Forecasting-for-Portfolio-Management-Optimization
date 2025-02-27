import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(
    filename='../logs/data_preprocessor.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info('****************************Logging started for Data Preprocessor module****************************')

class DataPreprocessor:
    """
    Class to clean and preprocess financial time series data.
    
    Attributes:
        data (pd.DataFrame): Input data to preprocess.
        cleaned_data (pd.DataFrame): Processed data.
    """
    
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
        logging.info("DataPreprocessor instance created.")
    
    def clean_data(self):
        """Cleans data by checking types, handling missing values, and ensuring consistency."""
        logging.info("Starting data cleaning.")
        try:
            # Ensure correct data types
            self.cleaned_data = self.data.copy()
            self.cleaned_data['Date'] = pd.to_datetime(self.cleaned_data['Date'])
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Handle missing values (interpolate for time series continuity)
            missing_count = self.cleaned_data.isnull().sum().sum()
            if missing_count > 0:
                self.cleaned_data = self.cleaned_data.interpolate(method='linear').ffill().bfill()
                logging.info("Handled %d missing values using interpolation.", missing_count)
            
            logging.info("Data cleaning completed. Basic stats: %s", self.cleaned_data.describe().to_dict())
            return self.cleaned_data
        except Exception as e:
            logging.error("Error during data cleaning: %s", e)
            raise
    
    def normalize_data(self):
        """Normalizes numerical columns using StandardScaler."""
        logging.info("Starting data normalization.")
        try:
            scaler = StandardScaler()
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            self.cleaned_data[numeric_cols] = scaler.fit_transform(self.cleaned_data[numeric_cols])
            logging.info("Data normalization completed.")
            return self.cleaned_data
        except Exception as e:
            logging.error("Error during normalization: %s", e)
            raise