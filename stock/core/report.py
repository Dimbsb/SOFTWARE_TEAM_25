from core.strategy import Strategy

import datetime
import matplotlib
import json
import os
import numpy as np
#
# -------------------------
# Report Class
# -------------------------
class Report:
    def __init__(self, predicted_price, loss, user, ticker, best_case, worst_case):
        self.user = user.name
        self.ticker = ticker
        self.predicted_price = predicted_price
        self.loss = loss
        self.best_case = best_case
        self.worst_case = worst_case

    def summary(self):
        return f"User: {self.user}\nTicker: {self.ticker}\nPredicted Close for tomorrow: {self.predicted_price:.2f}\nRange: {self.worst_case:.2f} – {self.best_case:.2f}\n"

