from flask import Flask, render_template, request
from core import User, Dataset, Analysis
import datetime

app = Flask(__name__)
user = User("Alice")

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    result = ""

    if request.method == "POST":
        if "ticker" in request.form:
            ticker = request.form["ticker"].upper()
            start = request.form.get("start_date", "2022-01-01")
            end = request.form.get("end_date", str(datetime.date.today()))
            dataset = Dataset(ticker, start, end)

            if not dataset.data.empty:
                user.upload_dataset(dataset)
                analysis = Analysis(dataset, user)
                analysis.run()
                result = f"{analysis.report.summary()}<br><b>Strategy:</b> {analysis.strategy.recommendation}"
            else:
                message = "Dataset not found or empty."

        elif "delete" in request.form:
            dataset_name = request.form["delete"]
            user.delete_dataset(dataset_name)
            message = f"Dataset '{dataset_name}' deleted."

    return render_template("index.html", datasets=user.datasets, message=message, result=result)

if __name__ == "__main__":
    app.run(debug=True)
