from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load(open("fraud_model.pkl", "rb"))
scaler = joblib.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return "Credit Card Fraud Detection API"

@app.route('/predict', methods=['POST'])
def predict():
   print("Received POST request")
    data = request.json['data']
    scaled_data = scaler.transform([np.array(data)])
    prediction = model.predict(scaled_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
