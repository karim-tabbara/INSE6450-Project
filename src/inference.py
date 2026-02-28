import numpy as np
from joblib import load
from scipy.sparse import hstack
from preprocessing import preprocess_text


def load_models():
    model = load('../models/logistic_regression_model.joblib')
    vectorizer = load('../models/tfidf_vectorizer.joblib')
    return model, vectorizer


def predict_email(text, model, vectorizer):
    processed_text = preprocess_text(text)
    text_features = vectorizer.transform([processed_text])
    
    final_features = text_features
    
    prediction = model.predict(final_features)
    return prediction[0]