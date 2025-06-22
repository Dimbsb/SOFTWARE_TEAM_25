import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib
matplotlib.use('Agg')
import datetime
from core.model import MLModel
from core.user import User
from core.strategy import Strategy
from core.dataset import Dataset
from core.analysis import Analysis
from core.report import Report

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
