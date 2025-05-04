from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Flask app setup
app = Flask(__name__, template_folder="templates")

# Load model (model.pkl is now inside Week5/)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_features = np.array([data])
        prediction = model.predict(final_features)
        return render_template("index.html", prediction_text=f"Prediction: {prediction[0]}")
    except Exception as e:
        return jsonify({"error": str(e)})

# Run app on Heroku port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
