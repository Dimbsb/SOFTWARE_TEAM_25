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
