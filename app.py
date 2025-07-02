code = """
from flask import Flask, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)


# Load the model and scaler
model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route('/')
def home():
    return "Credit Card Fraud Detection API"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    scaled_data = scaler.transform([np.array(data)])
    prediction = model.predict(scaled_data)
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
"""


# Save to file
with open("/content/drive/MyDrive/Credit Card Fraud Detection/app.py", "w") as f:
    f.write(code)