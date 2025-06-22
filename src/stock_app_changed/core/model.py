
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib


matplotlib.use('Agg')

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
