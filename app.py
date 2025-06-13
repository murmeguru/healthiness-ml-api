from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('health_model.pkl')

# ✅ Configure Gemini with API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------
# ✅ Health score prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['sugar'], data['fat'], data['salt'], data['energy']]])
    prediction = model.predict(features)[0]
    return jsonify({'health_score': round(prediction, 2)})

# ----------------------------
# ✅ Gemini-based alternatives route
@app.route('/get_alternatives', methods=['POST'])
def get_alternatives():
    try:
        data = request.get_json()
        product_name = data.get('product_name', '')

        prompt = f"Suggest 2 to 3 healthy and commonly available alternatives for: {product_name}. Keep it short and return in bullet list format."

        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
        response = model.generate_content(prompt)

        return jsonify({
            'alternatives': response.text.strip()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------
# ✅ Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
