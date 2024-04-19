from flask import Flask, request, render_template, jsonify
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import re

app = Flask(__name__, template_folder='templates')

# Load models and scaler
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
label_encoder = joblib.load('label_encoder.pkl')

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['sentence']
    if sentence:
        cleaned_sentence = clean_text(sentence)
        sentence_embedding = sentence_model.encode([cleaned_sentence])
        sentence_embedding = scaler.transform(sentence_embedding)
        prediction_prob = rf_model.predict_proba(sentence_embedding)[:, 1][0]
        numeric_prediction = rf_model.predict(sentence_embedding)[0]
        decoded_prediction = label_encoder.inverse_transform([numeric_prediction])[0]
        return jsonify(prediction=decoded_prediction, probability=float(prediction_prob))
    else:
        return jsonify({"error": "No sentence provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
