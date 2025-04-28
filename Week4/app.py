from flask import Flask, request, render_template
import joblib

# 🔧 Initialize Flask
app = Flask(__name__)

# 📦 Load your saved model
model = joblib.load('model.pkl')

# 🏠 Home route
@app.route('/')
def home():
    return render_template('index.html')

# 🔮 Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        data = [float(x) for x in request.form.values()]
        prediction = model.predict([data])
        return render_template('index.html', prediction_text=f'Predicted Class: {prediction[0]}')
    except:
        return render_template('index.html', prediction_text="⚠️ Please enter valid numbers.")

# ▶️ Run the app
if __name__ == '__main__':
    app.run(debug=True)
