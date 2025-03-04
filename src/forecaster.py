import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(
    filename='../logs/forecaster.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info('Logging started for Forecaster module')


class Forecaster:
    """
    Class to develop and evaluate time series forecasting models for stock prices.

    Attributes:
        ticker (str): Ticker symbol (e.g., 'TSLA').
        data (pd.DataFrame): Input time series data (e.g., TSLA Close prices).
        train (pd.Series): Training data.
        test (pd.Series): Testing data.
    """

    def __init__(self, data, ticker, test_size=0.2):
        self.ticker = ticker
        self.data = data['Close'].dropna()  # Focus on Close prices
        self.test_size = test_size
        self.train = self.data.iloc[:int(len(self.data) * (1 - test_size))]
        self.test = self.data.iloc[int(len(self.data) * (1 - test_size)):]
        logging.info("Forecaster instance created for %s. Train size: %d, Test size: %d",
                     self.ticker, len(self.train), len(self.test))

    def forecast_arima(self, optimize=True):
        logging.info("Starting ARIMA forecasting for %s", self.ticker)
        try:
            if optimize:
                model = auto_arima(self.train, start_p=0, start_q=0, max_p=5, max_q=5, d=1,
                                   seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
                order = model.order
            else:
                order = (1, 1, 1)
            arima_model = ARIMA(self.train, order=order)
            arima_fit = arima_model.fit()
            forecast = arima_fit.forecast(steps=len(self.test))
            logging.info(
                "ARIMA forecast completed for %s with order: %s", self.ticker, order)
            return self._evaluate_forecast(forecast, "ARIMA")
        except Exception as e:
            logging.error(
                "Error in ARIMA forecasting for %s: %s", self.ticker, e)
            raise

    def forecast_sarima(self, optimize=True):
        logging.info("Starting SARIMA forecasting for %s", self.ticker)
        try:
            if optimize:
                model = auto_arima(self.train, start_p=0, start_q=0, max_p=5, max_q=5, d=1,
                                   seasonal=True, m=252, start_P=0, start_Q=0, max_P=2, max_Q=2,
                                   trace=False, error_action='ignore', suppress_warnings=True)
                order = model.order
                seasonal_order = model.seasonal_order
            else:
                order = (1, 1, 1)
                seasonal_order = (1, 0, 1, 252)
            sarima_model = SARIMAX(
                self.train, order=order, seasonal_order=seasonal_order)
            sarima_fit = sarima_model.fit(disp=False)
            forecast = sarima_fit.forecast(steps=len(self.test))
            logging.info("SARIMA forecast completed for %s with order: %s, seasonal_order: %s",
                         self.ticker, order, seasonal_order)
            return self._evaluate_forecast(forecast, "SARIMA")
        except Exception as e:
            logging.error(
                "Error in SARIMA forecasting for %s: %s", self.ticker, e)
            raise

    # def forecast_lstm(self, look_back=60, epochs=50, batch_size=32):
    #     logging.info("Starting LSTM forecasting for %s (GPU recommended)", self.ticker)
    #     try:
    #         scaler = MinMaxScaler()
    #         scaled_data = scaler.fit_transform(self.data.values.reshape(-1, 1))
    #         train_scaled = scaled_data[:int(len(scaled_data) * (1 - self.test_size))]
    #         test_scaled = scaled_data[int(len(scaled_data) * (1 - self.test_size)):]

    #         X_train, y_train = self._create_sequences(train_scaled, look_back)
    #         X_test, y_test = self._create_sequences(test_scaled, look_back)

    #         model = Sequential()
    #         model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    #         model.add(LSTM(50))
    #         model.add(Dense(1))
    #         model.compile(optimizer='adam', loss='mse')

    #         model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    #         forecast_scaled = model.predict(X_test, verbose=0)
    #         forecast = scaler.inverse_transform(forecast_scaled)

    #         forecast_series = pd.Series(forecast.flatten(), index=self.test.index[-len(forecast):])
    #         logging.info("LSTM forecast completed for %s with look_back: %d", self.ticker, look_back)
    #         return self._evaluate_forecast(forecast_series, "LSTM")
    #     except Exception as e:
    #         logging.error("Error in LSTM forecasting for %s: %s", self.ticker, e)
    #         raise

    def _create_sequences(self, data, look_back):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    def _evaluate_forecast(self, forecast, model_name):
        logging.info("Evaluating %s forecast for %s", model_name, self.ticker)
        try:
            mae = mean_absolute_error(self.test[-len(forecast):], forecast)
            rmse = np.sqrt(mean_squared_error(
                self.test[-len(forecast):], forecast))
            mape = np.mean(np.abs(
                (self.test[-len(forecast):] - forecast) / self.test[-len(forecast):])) * 100
            metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
            logging.info("%s metrics for %s: %s",
                         model_name, self.ticker, metrics)
            return forecast, metrics
        except Exception as e:
            logging.error("Error evaluating forecast for %s: %s",
                          self.ticker, e)
            raise
