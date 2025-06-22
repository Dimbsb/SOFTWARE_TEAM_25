from core.model import MLModel
from core.user import User
from core.strategy import Strategy

from core.report import Report
import matplotlib
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

    def plot_prediction(self, output_path="static/plot.png"):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        close_prices = self.model.data['Close'].values
        if len(close_prices) < 10:
            print("Not enough data to plot last 10 days.")
            return

        # Πάρε τις 10 τελευταίες τιμές
        last_10 = close_prices[-10:]
        last_real_price = float(last_10[-1])

        # Πρόβλεψη
        predicted_price = float(self.model.predict_last_day())

        # Δημιουργία x-axis με ημερομηνίες
        last_dates = self.model.data.index[-10:]
        next_date = pd.to_datetime(last_dates[-1]) + pd.Timedelta(days=1)
        all_dates = list(last_dates) + [next_date]

        y_values = list(last_10) + [predicted_price]

        color = 'green' if predicted_price > last_real_price else 'red'

        # Plot
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#232526')
        ax.set_facecolor('#2c2f33')

        ax.plot(all_dates[:-1], last_10, color='#ffd700', label="Ιστορικά Δεδομένα", linewidth=2.2)
        ax.plot([all_dates[-2], all_dates[-1]], [last_real_price, predicted_price],
                color=color, linestyle='--', linewidth=2.5, marker='o', label="Πρόβλεψη")

        ax.set_title(f"Πρόβλεψη Τιμής Μετοχής ({self.dataset.ticker})", fontsize=14, color='#fffbe6')
        ax.set_xlabel("Ημερομηνία", fontsize=12, color='#ffd700')
        ax.set_ylabel("Τιμή ($)", fontsize=12, color='#ffd700')
        ax.tick_params(colors='#fffbe6')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend().get_texts()[0].set_color('#fffbe6')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close()

    def run(self):
        self.model.preprocess()
        self.model.build_model()
        self.model.train(epochs=3)  # For demo
        predicted_price = self.model.predict_last_day()
        last_price = self.dataset.data['Close'].values[-1]
        self.strategy = Strategy(predicted_price, last_price)
        self.report = Report(predicted_price, self.model.model.history.history['loss'][-1], self.user, self.dataset.ticker)
        self.plot_prediction()


    def show_results(self):
        print("===== Analysis Report =====")
        print(self.report.summary())
        print("===== Strategy Suggestion =====")
        print(self.strategy.recommendation)
