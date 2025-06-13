from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import openai

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('health_model.pkl')

# ----------------------------
# ✅ Existing health score prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['sugar'], data['fat'], data['salt'], data['energy']]])
    prediction = model.predict(features)[0]
    return jsonify({'health_score': round(prediction, 2)})

# ----------------------------
# ✅ NEW: ChatGPT-based alternatives route
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/get_alternatives', methods=['POST'])
def get_alternatives():
    data = request.get_json()
    product_name = data.get('product_name', '')

    prompt = f"Suggest 2 or 3 healthier food or drink alternatives for: {product_name}. Respond in this format: 'Healthier alternatives: ...'"

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        reply = response.choices[0].message.content.strip()
        return jsonify({"alternatives": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# ✅ App run configuration
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
