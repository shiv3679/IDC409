from flask import Flask, request, jsonify
import joblib
from sentence_transformers import SentenceTransformer
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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'sentence' in data:
        # Preprocess the sentence
        cleaned_sentence = clean_text(data['sentence'])
        # Generate embeddings
        sentence_embedding = sentence_model.encode([cleaned_sentence])
        # Make prediction
        numeric_prediction = lr_model.predict(sentence_embedding)
        # Decode the numeric prediction back to original label
        decoded_prediction = label_encoder.inverse_transform(numeric_prediction)[0]
        return jsonify(prediction=decoded_prediction)
    else:
        return jsonify({"error": "No sentence provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
