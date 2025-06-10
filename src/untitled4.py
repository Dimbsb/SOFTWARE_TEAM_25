import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------
# User Class
# -------------------------
class User:
    def __init__(self, name):
        self.name = name
        self.datasets = []

    def upload_dataset(self, dataset):
        self.datasets.append(dataset)
        print(f"{self.name} uploaded dataset: {dataset.name}")

# -------------------------
# Dataset Class
# -------------------------
class Dataset:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.name = f"{ticker} Stock Data"
        self.data = yf.download(ticker, start=start_date, end=end_date)

# -------------------------
# Analysis Class
# -------------------------
class Analysis:
    def __init__(self, dataset):
        self.dataset = dataset

    def compute_daily_return(self):
        self.dataset.data['Daily Return'] = self.dataset.data['Close'].pct_change()
        print("Computed daily returns.")

    def plot_daily_return(self):
        self.dataset.data['Daily Return'].plot(figsize=(10, 4), title="Daily Return")
        plt.show()

# -------------------------
# MLModel Class
# -------------------------
class MLModel:
    def __init__(self, dataset):
        self.data = dataset.data
        self.scaler = MinMaxScaler()
        self.model = None
        self.X_train = None
        self.y_train = None

    def preprocess(self):
        close_prices = self.data[['Close']].values
        scaled_data = self.scaler.fit_transform(close_prices)

        sequence_length = 60
        X = []
        y = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        print("Preprocessing complete.")

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print("Model built.")

    def train(self, epochs=1, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)
        print("Training complete.")

    def predict_last_day(self):
        last_60_days = self.data['Close'].values[-60:]
        scaled = self.scaler.transform(last_60_days.reshape(-1, 1))
        X_test = np.reshape(scaled, (1, 60, 1))
        predicted_price = self.model.predict(X_test)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        print(f"Predicted next close price for the next day is: {predicted_price[0][0]:.2f}")
        return predicted_price[0][0]

# -------------------------
# Execution Flow
# -------------------------
def main():
    # Create user
    user = User("Alice")

    # Download dataset
    dataset = Dataset(ticker="AAPL", start_date="2022-01-10", end_date="2025-05-31")
    user.upload_dataset(dataset)

    # Analysis
    analysis = Analysis(dataset)
    analysis.compute_daily_return()
    analysis.plot_daily_return()

    # ML Model
    model = MLModel(dataset)
    model.preprocess()
    model.build_model()
    model.train(epochs=5)  # keep low for demo
    model.predict_last_day()

if __name__ == "__main__":
    main()