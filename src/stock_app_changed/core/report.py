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

