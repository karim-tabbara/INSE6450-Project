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
    # print(f"Raw Data shape: {df.shape}")
    # print(f"Raw Data Label distribution:\n{df.label.value_counts()}")

    # Extract features
    unprocessed_text = df["message"]
    character_count = unprocessed_text.str.len()
    word_count = unprocessed_text.str.split().apply(len)
    question_count = unprocessed_text.str.count(r"\?")
    exclamation_count = unprocessed_text.str.count("!")

    numeric_features = np.column_stack([
        character_count,
        word_count,
        question_count,
        exclamation_count
    ])
    # print("=================================================")
    # print(f"Numeric features shape: {numeric_features.shape}")
    # print(numeric_features[:3])

    # Preprocess text
    processed_text = unprocessed_text.apply(preprocess_text)

    X_text = processed_text
    y = df["label"]

    # Training set: 70%
    # Validation set: 15%
    # Test set: 15%
    X_train_text, X_temp_text, y_train, y_temp, num_train, num_temp = train_test_split(
        X_text, y, numeric_features,
        test_size=0.30, 
        stratify=y,
        random_state=42
    )
    
    X_val_text, X_test_text, y_val, y_test, num_val, num_test = train_test_split(
        X_temp_text, y_temp, num_temp, 
        test_size=0.50, 
        stratify=y_temp, 
        random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    scaler = StandardScaler(with_mean=False)
    num_train_scaled = scaler.fit_transform(num_train)
    num_val_scaled = scaler.transform(num_val)
    num_test_scaled = scaler.transform(num_test)

    X_train_final = hstack([X_train_tfidf, num_train_scaled])
    X_val_final = hstack([X_val_tfidf, num_val_scaled])
    X_test_final = hstack([X_test_tfidf, num_test_scaled])

    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, vectorizer, scaler
