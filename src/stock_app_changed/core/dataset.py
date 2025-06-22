import yfinance as yf


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
