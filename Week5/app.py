from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder="Week5/templates")
model = joblib.load("Week5/model.pkl")

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

if __name__ == "__main__":
    app.run(debug=True)
