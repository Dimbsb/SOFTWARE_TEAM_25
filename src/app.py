from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

# Dummy prediction function, replace with your model's prediction logic
def predict_next_close_price():
    # In practice, get your model prediction here
    predicted_price = 175.32
    next_date = datetime.now().date()
    return next_date, predicted_price

@app.route('/predict')
def predict():
    date, price = predict_next_close_price()
    return jsonify({
        "date": str(date),
        "predicted_close_price": price
    })

if __name__ == '__main__':
    app.run(debug=True)
