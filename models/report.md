# Project Report: Question Detection Model

## Overview
This report outlines the development and deployment of a machine learning model designed to identify whether sentences from video transcripts are questions. The project includes data preprocessing, model training, API development, and dockerization for deployment.

## Data Preprocessing
Data preprocessing involved cleaning text data, encoding categorical labels, and preparing the data for model training. 

### Steps:
1. **Text Cleaning**: Lowercased all text, removed punctuation, and stripped whitespaces.
2. **Label Encoding**: Converted text labels into integers using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

data.drop(columns=data.columns[0], inplace=True)
data['sentence'] = data['sentence'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])
