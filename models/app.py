from flask import Flask, request, jsonify, render_template
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import re

app = Flask(__name__)

# Load the trained logistic regression model
lr_model = joblib.load('logistic_regression_model.pkl')

# Load the sentence embedding model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the LabelEncoder
label_encoder = joblib.load('label_encoder.pkl')

def clean_text(text):
    """Preprocess text by lowercasing and removing non-alphanumeric characters"""
    return re.sub(r'[^\w\s]', '', text.lower().strip())

@app.route('/', methods=['GET'])
def index():
    # Serve the HTML form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data sent to the endpoint
    if 'sentence' in data:
        sentence = data['sentence']
        cleaned_sentence = clean_text(sentence)
        sentence_embedding = sentence_model.encode([cleaned_sentence])
        prediction_prob = lr_model.predict_proba(sentence_embedding)[:, 1][0]
        numeric_prediction = lr_model.predict(sentence_embedding)[0]
        decoded_prediction = label_encoder.inverse_transform([numeric_prediction])[0]
        return jsonify(prediction=decoded_prediction, probability=float(prediction_prob))
    else:
        return jsonify({"error": "No sentence provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
