import numpy as np
from joblib import load
from scipy.sparse import hstack
import model
from preprocessing import preprocess_text
import psutil
import os
import time
import math


def load_models(version='v1'):
    if version == 'v1':
        model = load('../models/logistic_regression_model.joblib')
        vectorizer = load('../models/tfidf_vectorizer.joblib')
    elif version == 'v2':
        model = load('../models/logistic_regression_model_v2.joblib')
        vectorizer = load('../models/tfidf_vectorizer_v2.joblib')
    elif version == 'CL':
        model = load('../models/logistic_regression_model_CL.joblib')
        vectorizer = load('../models/tfidf_vectorizer_CL.joblib')
    elif version == 'HITL':
        model = load('../models/logistic_regression_model_HITL.joblib')
        vectorizer = load('../models/tfidf_vectorizer_HITL.joblib')
        
    return model, vectorizer


def predict_email(text, model, vectorizer):
    
    if not text or not text.strip():
        return "ABSTAIN", 0.0
    
    processed_text = preprocess_text(text)
    text_features = vectorizer.transform([processed_text])
    
    final_features = text_features
    
    probabilities = model.predict_proba(final_features)
    max_prob = probabilities.max()
    predicted_class = model.classes_[probabilities.argmax()]

    if max_prob < 0.65:
        return "ABSTAIN", max_prob

    return predicted_class, max_prob



def measure_single_inference_metrics(texts, model, vectorizer, runs=1000, warmup=50):
    latencies = []

    process = psutil.Process(os.getpid())

    # Warmup runs
    for _ in range(warmup):
        text = texts[np.random.randint(len(texts))]
        predict_email(text, model, vectorizer)
    
    # Timed runs
    for _ in range(runs):
        text = texts[np.random.randint(len(texts))]
        
        start_time = time.perf_counter()
        predict_email(text, model, vectorizer)
        latency = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)
    
    memory_single_mb = process.memory_info().rss / (1024 * 1024)

    latencies = np.array(latencies)
    latency_single_p50 = np.percentile(latencies, 50)
    latency_single_p90 = np.percentile(latencies, 90)

    return latency_single_p50, latency_single_p90, memory_single_mb



def measure_batch_inference_metrics(texts, model, vectorizer, batch_size=32, runs=100, warmup=10):
    latencies = []

    process = psutil.Process(os.getpid())

    # Warmup runs
    for _ in range(warmup):
        batch = np.random.choice(texts, batch_size, replace=False)
        X_batch = vectorizer.transform(batch)
        model.predict(X_batch)
    
    # Timed runs
    start_time_total = time.perf_counter()

    for _ in range(runs):
        batch = np.random.choice(texts, batch_size, replace=False)
        start_time = time.perf_counter()
        X_batch = vectorizer.transform(batch)
        model.predict(X_batch)
        latency = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)
    
    time_total = (time.perf_counter() - start_time_total) * 1000  # Convert to milliseconds
    memory_batch_mb = process.memory_info().rss / (1024 * 1024)

    total_samples = batch_size * runs
    throughput = math.floor(total_samples / (time_total / 1000))

    latencies = np.array(latencies)
    latency_batch_p50 = np.percentile(latencies, 50)
    latency_batch_p90 = np.percentile(latencies, 90)

    return latency_batch_p50, latency_batch_p90, memory_batch_mb, throughput