from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('health_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['sugar'], data['fat'], data['salt'], data['energy']]])
    prediction = model.predict(features)[0]
    return jsonify({'health_score': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
