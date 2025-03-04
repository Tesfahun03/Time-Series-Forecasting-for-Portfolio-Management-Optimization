import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
        data (pd.DataFrame): Input time series data (e.g., TSLA Close prices).
        train (pd.Series): Training data.
        test (pd.Series): Testing data.
    """

    def __init__(self, data, test_size=0.2):
        self.data = data['Close'].dropna()  # Focus on TSLA Close prices
        self.test_size = test_size
        self.train = self.data.iloc[:int(len(self.data) * (1 - test_size))]
        self.test = self.data.iloc[int(len(self.data) * (1 - test_size)):]
        logging.info("Forecaster instance created. Train size: %d, Test size: %d", len(
            self.train), len(self.test))

    def forecast_arima(self, optimize=True):
        """Forecasts using ARIMA with optional parameter optimization."""
        logging.info("Starting ARIMA forecasting")
        try:
            if optimize:
                model = auto_arima(self.train, start_p=0, start_q=0, max_p=5, max_q=5, d=1,
                                   seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
                order = model.order
            else:
                order = (1, 1, 1)  # Default if not optimizing
            arima_model = ARIMA(self.train, order=order)
            arima_fit = arima_model.fit()
            forecast = arima_fit.forecast(steps=len(self.test))
            logging.info("ARIMA forecast completed with order: %s", order)
            return self._evaluate_forecast(forecast, "ARIMA")
        except Exception as e:
            logging.error("Error in ARIMA forecasting: %s", e)
            raise

    def forecast_sarima(self, optimize=True):
        """Forecasts using SARIMA with optional parameter optimization."""
        logging.info("Starting SARIMA forecasting")
        try:
            if optimize:
                model = auto_arima(self.train, start_p=0, start_q=0, max_p=5, max_q=5, d=1,
                                   seasonal=True, m=252, start_P=0, start_Q=0, max_P=2, max_Q=2,
                                   trace=False, error_action='ignore', suppress_warnings=True)
                order = model.order
                seasonal_order = model.seasonal_order
            else:
                order = (1, 1, 1)
                seasonal_order = (1, 0, 1, 252)  # Annual seasonality
            sarima_model = SARIMAX(
                self.train, order=order, seasonal_order=seasonal_order)
            sarima_fit = sarima_model.fit(disp=False)
            forecast = sarima_fit.forecast(steps=len(self.test))
            logging.info(
                "SARIMA forecast completed with order: %s, seasonal_order: %s", order, seasonal_order)
            return self._evaluate_forecast(forecast, "SARIMA")
        except Exception as e:
            logging.error("Error in SARIMA forecasting: %s", e)
            raise

    # def forecast_lstm(self, look_back=60, epochs=50, batch_size=32):
    #     """Forecasts using LSTM. Note: GPU recommended for faster training."""
    #     logging.info(
    #         "Starting LSTM forecasting (GPU recommended for large datasets)")
    #     try:
    #         # Scale data
    #         scaler = MinMaxScaler()
    #         scaled_data = scaler.fit_transform(self.data.values.reshape(-1, 1))
    #         train_scaled = scaled_data[:int(
    #             len(scaled_data) * (1 - self.test_size))]
    #         test_scaled = scaled_data[int(
    #             len(scaled_data) * (1 - self.test_size)):]

    #         # Prepare sequences
    #         X_train, y_train = self._create_sequences(train_scaled, look_back)
    #         X_test, y_test = self._create_sequences(test_scaled, look_back)

    #         # Build LSTM model
    #         model = Sequential()
    #         model.add(LSTM(50, return_sequences=True,
    #                   input_shape=(look_back, 1)))
    #         model.add(LSTM(50))
    #         model.add(Dense(1))
    #         model.compile(optimizer='adam', loss='mse')

    #         # Train model
    #         model.fit(X_train, y_train, epochs=epochs,
    #                   batch_size=batch_size, verbose=0)
    #         forecast_scaled = model.predict(X_test, verbose=0)
    #         forecast = scaler.inverse_transform(forecast_scaled)

    #         # Align forecast with test set
    #         forecast_series = pd.Series(
    #             forecast.flatten(), index=self.test.index[-len(forecast):])
    #         logging.info(
    #             "LSTM forecast completed with look_back: %d", look_back)
    #         return self._evaluate_forecast(forecast_series, "LSTM")
    #     except Exception as e:
    #         logging.error("Error in LSTM forecasting: %s", e)
    #         raise

    def _create_sequences(self, data, look_back):
        """Helper to create sequences for LSTM."""
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    def _evaluate_forecast(self, forecast, model_name):
        """Evaluates forecast performance with MAE, RMSE, MAPE."""
        logging.info("Evaluating %s forecast", model_name)
        try:
            mae = mean_absolute_error(self.test[-len(forecast):], forecast)
            rmse = np.sqrt(mean_squared_error(
                self.test[-len(forecast):], forecast))
            mape = np.mean(np.abs(
                (self.test[-len(forecast):] - forecast) / self.test[-len(forecast):])) * 100
            metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
            logging.info("%s metrics: %s", model_name, metrics)
            return forecast, metrics
        except Exception as e:
            logging.error("Error evaluating forecast: %s", e)
            raise
