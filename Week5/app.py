from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['sepal length (cm)']),
        float(request.form['sepal width (cm)']),
        float(request.form['petal length (cm)']),
        float(request.form['petal width (cm)'])
    ]
    prediction = model.predict([features])
    return render_template('index.html', prediction_text=f'Predicted Class: {prediction[0]}')

# ✅ Add this new route for API
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    features = [list(data.values())]
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
