import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def load_and_prepare_data(path):
    df = pd.read_csv(path)

    unprocessed_text = df["message"]

    # Preprocess text
    processed_text = unprocessed_text.apply(preprocess_text)

    X_text = processed_text
    y = df["label"]

    # Training set: 70%
    # Validation set: 15%
    # Test set: 15%

    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X_text, y,
        test_size=0.30, 
        stratify=y,
        random_state=42
    )
    
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp,
        test_size=0.50, 
        stratify=y_temp, 
        random_state=42
    )

    # vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    X_train_final = X_train_tfidf
    X_val_final = X_val_tfidf
    X_test_final = X_test_tfidf

    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, vectorizer
