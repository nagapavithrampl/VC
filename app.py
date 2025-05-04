from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Map numeric predictions to flower class labels
label_map = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}

# Set correct template folder path
app = Flask(__name__, template_folder="templates")

# Load model from Week5 folder
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
        label = label_map.get(prediction[0], "Unknown")
        return render_template("index.html", prediction_text=f"Prediction: {label}")
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
