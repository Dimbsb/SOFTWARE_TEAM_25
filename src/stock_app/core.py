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
        self.analyses = []

    def upload_dataset(self, dataset):
        if dataset.data is not None and not dataset.data.empty:
            self.datasets.append(dataset)
            print(f"{self.name} uploaded dataset: {dataset.name}")
        else:
            print("Dataset is empty. Upload failed.")

    def delete_dataset(self, dataset_name):
        for ds in self.datasets:
            if ds.name == dataset_name:
                self.datasets.remove(ds)
                print(f"Dataset '{dataset_name}' deleted.")
                return
        print(f"No dataset named '{dataset_name}' found.")

# -------------------------
# Dataset Class
# -------------------------
class Dataset:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.name = f"{ticker} Stock Data"
        self.data = yf.download(ticker, start=start_date, end=end_date)
        if self.data.empty:
            print("Warning: Dataset is empty.")

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
        self.trained = False

    def preprocess(self):
        close_prices = self.data[['Close']].values
        scaled_data = self.scaler.fit_transform(close_prices)

        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i - sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        print("Preprocessing complete.")

    def build_model(self):
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
            LSTM(units=50),
            Dense(units=1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print("Model built.")

    def train(self, epochs=5, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)
        self.trained = True
        print("Training complete.")

    def predict_last_day(self):
        last_60_days = self.data['Close'].values[-60:]
        scaled = self.scaler.transform(last_60_days.reshape(-1, 1))
        X_test = np.reshape(scaled, (1, 60, 1))
        predicted_price = self.model.predict(X_test)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        return predicted_price[0][0]

# -------------------------
# Strategy Class
# -------------------------
class Strategy:
    def __init__(self, predicted_price, last_real_price):
        self.predicted_price = predicted_price
        self.last_real_price = last_real_price
        self.recommendation = self.generate_strategy()

 
    def generate_strategy(self):
        change = ((self.predicted_price - self.last_real_price) / self.last_real_price) * 100
        change = float(change)  # Ensure it's a Python float
        if change > 2:
            return f"Buy — Expected increase of {change:.2f}%"
        elif change < -2:
            return f"Sell — Expected decrease of {change:.2f}%"
        else:
            return f"Hold — Change is negligible ({change:.2f}%)"
 

# -------------------------
# Report Class
# -------------------------
class Report:
    def __init__(self, predicted_price, loss, user, ticker):
        self.user = user.name
        self.ticker = ticker
        self.predicted_price = predicted_price
        self.loss = loss

    def summary(self):
        return f"User: {self.user}\nTicker: {self.ticker}\nPredicted Close: {self.predicted_price:.2f}\nTraining Loss: {self.loss:.4f}"

# -------------------------
# Analysis Class
# -------------------------
class Analysis:
    def __init__(self, dataset, user):
        self.dataset = dataset
        self.user = user
        self.model = MLModel(dataset)
        self.strategy = None
        self.report = None
        self.user.analyses.append(self)

    def run(self):
        self.model.preprocess()
        self.model.build_model()
        self.model.train(epochs=3)  # For demo
        predicted_price = self.model.predict_last_day()
        last_price = self.dataset.data['Close'].values[-1]
        self.strategy = Strategy(predicted_price, last_price)
        self.report = Report(predicted_price, self.model.model.history.history['loss'][-1], self.user, self.dataset.ticker)

    def show_results(self):
        print("===== Analysis Report =====")
        print(self.report.summary())
        print("===== Strategy Suggestion =====")
        print(self.strategy.recommendation)

# -------------------------
# Admin Class
# -------------------------
class Admin:
    def __init__(self):
        self.users = []

    def register_user(self, user):
        self.users.append(user)
        print(f"Admin: Registered user '{user.name}'.")

    def delete_user(self, username):
        self.users = [u for u in self.users if u.name != username]
        print(f"Admin: Deleted user '{username}' if existed.")

    def list_users(self):
        print("Admin: Current users:")
        for u in self.users:
            print(f"- {u.name}")

# -------------------------
# Execution Flow
# -------------------------
def main():
    admin = Admin()

    # Δημιουργία χρήστη
    user = User("Alice")
    admin.register_user(user)

    # Εισαγωγή δεδομένων
    dataset = Dataset(ticker="AAPL", start_date="2022-01-01", end_date="2025-06-01")
    user.upload_dataset(dataset)

    # Διαγραφή dataset (παράδειγμα)
    # user.delete_dataset("AAPL Stock Data")

    # Ανάλυση
    analysis = Analysis(dataset, user)
    analysis.run()
    analysis.show_results()

if __name__ == "__main__":
    main()
