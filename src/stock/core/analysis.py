
from core.model import MLModel
from core.user import User
from core.strategy import Strategy
from core.report import Report
import datetime
import matplotlib
import json
import os
import numpy as np
import zipfile
matplotlib.use('Agg')

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

    def plot_prediction(self, worst_case, best_case, long_term_forecast, output_path="static/plot.png"):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
 
        close_prices = self.model.data['Close'].values
        if len(close_prices) < 60:
            print("Not enough data to plot.")
            return

        # Short-term για τη γραφική
        last_60 = close_prices[-60:]
        last_real_price = float(last_60[-1])
        predicted_price = float(self.model.predict_last_day())

        # Ημερομηνίες
        last_dates = self.model.data.index[-60:]
        next_date = pd.to_datetime(last_dates[-1]) + pd.Timedelta(days=1)

        # 🛠️ Ensure long_term_forecast is a NumPy array
        if hasattr(long_term_forecast, "values"):
            long_term_forecast = long_term_forecast.values
        long_term_forecast = np.array(long_term_forecast).flatten()
        future_dates = pd.date_range(start=next_date, periods=len(long_term_forecast))

        color = 'green' if predicted_price > last_real_price else 'red'




        last_price = self.model.data['Close'].values[-1]

        # Υπολογισμός rolling std σε τιμές κλεισίματος
        rolling_std = self.model.data['Close'].pct_change().rolling(window=20).std().iloc[-1]
        std_dev = predicted_price * float(rolling_std)

        long_term_forecast = np.asarray(long_term_forecast).flatten()
        std_dev = float(std_dev)
        forecast_best = long_term_forecast + std_dev
        forecast_worst = long_term_forecast - std_dev

        # Plot
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#232526')
        ax.set_facecolor('#2c2f33')

        # Ιστορικά
        ax.plot(last_dates, last_60, color='#ffd700', label="Ιστορικά Δεδομένα", linewidth=2.2)

        # Short-term πρόβλεψη (μία μέρα)
        ax.plot([last_dates[-1], next_date], [last_real_price, predicted_price],
                color=color, linestyle='--', linewidth=2.5, marker='o', label="Short-Term Prediction")

        # Long-term πρόβλεψη
        ax.plot(future_dates, long_term_forecast, color=color, linestyle='-', linewidth=2.5, marker='o', label="Long-Term Forecast")
        ax.plot(future_dates, forecast_best, color='lime', linestyle='--', linewidth=1.5, label="Best Case")
        ax.plot(future_dates, forecast_worst, color='orange', linestyle='--', linewidth=1.5, label="Worst Case")
        ax.fill_between(future_dates, forecast_worst, forecast_best, color='gray', alpha=0.2, label="Uncertainty Band")

        ax.set_title(f"Stock Price Forecast ({self.dataset.ticker})", fontsize=14, color='#fffbe6')
        ax.set_xlabel("Ημερομηνία", fontsize=12, color='#ffd700')
        ax.set_ylabel("Τιμή ($)", fontsize=12, color='#ffd700')
        ax.tick_params(colors='#fffbe6')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close()



        # Δημιουργία DataFrame για export
        graph_df = pd.DataFrame({
            "Date": list(last_dates) + list(future_dates),
            "Close": list(last_60) + [np.nan] * len(long_term_forecast),
            "Forecast": [np.nan] * len(last_60) + list(long_term_forecast),
            "BestCase": [np.nan] * len(last_60) + list(forecast_best),
            "WorstCase": [np.nan] * len(last_60) + list(forecast_worst),
        })

        graph_df.to_csv("static/graph_data.csv", index=False)

    def save_results_to_json(self, output_path="static/results.json"):
        results = {
            "ticker": self.dataset.ticker,
            "predicted_price": round(float(self.report.predicted_price), 2),
            "best_case": round(float(self.report.best_case), 2),
            "worst_case": round(float(self.report.worst_case), 2),
            "loss": round(float(self.report.loss), 6),
            "recommendation": self.strategy.recommendation,
            "date": datetime.date.today().isoformat()
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)



    def run(self):
        self.model.preprocess()
        self.model.build_model()
        self.model.train(epochs=3)

        today = datetime.date.today()
        end = datetime.datetime.strptime(self.dataset.end_date, "%Y-%m-%d").date()
        steps = max(1, (end - today).days)

        predicted_price = self.model.predict_last_day()
        long_term_forecast = self.model.predict_until(steps)
        if hasattr(long_term_forecast, "values"):
            long_term_forecast = long_term_forecast.values
        long_term_forecast = np.array(long_term_forecast).flatten()

        loss = self.model.model.history.history['loss'][-1]


        rolling_std = self.model.data['Close'].pct_change().rolling(window=20).std().iloc[-1]
        std_dev = predicted_price * float(rolling_std)

        print("long_term_forecast shape:", long_term_forecast.shape)
        print("std_dev:", std_dev)

        best_case = predicted_price + std_dev
        worst_case = predicted_price - std_dev

        last_price = self.dataset.data['Close'].values[-1]
        self.strategy = Strategy(predicted_price, last_price)
        self.report = Report(predicted_price, loss, self.user, self.dataset.ticker, best_case, worst_case)

        self.plot_prediction(worst_case, best_case, long_term_forecast)
        self.save_text_summary()
        self.create_zip_report()

    def show_results(self):
        print("===== Analysis Report =====")
        print(self.report.summary())
        print("===== Strategy Suggestion =====")
        print(self.strategy.recommendation)

    def save_text_summary(self, output_path="static/report.txt"):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("===== Stock Forecast Report =====\n")
            f.write(f"Ticker: {self.dataset.ticker}\n")
            f.write(f"Date: {datetime.date.today().isoformat()}\n")
            f.write(f"Predicted Close: {round(self.report.predicted_price, 2)}\n")
            f.write(f"Best Case: {round(self.report.best_case, 2)}\n")
            f.write(f"Worst Case: {round(self.report.worst_case, 2)}\n")
            f.write(f"Loss: {round(self.report.loss, 6)}\n")
            f.write(f"Recommendation: {self.strategy.recommendation}\n")



    def create_zip_report(self, zip_path="static/forecast_report.zip"):
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write("static/plot.png", arcname="plot.png")
            zipf.write("static/graph_data.csv", arcname="graph_data.csv")
            zipf.write("static/report.txt", arcname="report.txt")

