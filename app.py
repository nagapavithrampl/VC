from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="templates")
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        prediction = model.predict([data])
        return render_template("index.html", prediction_text=f"Prediction: {prediction[0]}")
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
