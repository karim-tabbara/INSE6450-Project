import numpy as np
from joblib import load
from scipy.sparse import hstack
from preprocessing import preprocess_text


def load_models():
    model = load('../models/logistic_regression_model.joblib')
    vectorizer = load('../models/tfidf_vectorizer.joblib')
    scaler = load('../models/standard_scaler.joblib')
    return model, vectorizer, scaler


def predict_email(text, model, vectorizer, scaler):
    processed_text = preprocess_text(text)
    text_features = vectorizer.transform([processed_text])
    
    character_count = len(text)
    word_count = len(text.split())
    question_count = text.count("?")
    exclamation_count = text.count("!")
    
    numeric_features = np.array([[character_count, word_count, question_count, exclamation_count]])
    numeric_features_scaled = scaler.transform(numeric_features)
    
    final_features = hstack([text_features, numeric_features_scaled])
    
    prediction = model.predict(final_features)
    return prediction[0]