from joblib import dump
from sklearn.model_selection import train_test_split
from inference import load_models, measure_batch_inference_metrics, measure_single_inference_metrics
from model import evaluate_model, train_model
import pandas as pd
from preprocessing import preprocess_text
from stress_tests import character_noise
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('../data/RawData.csv')
messages = df["message"]
labels = df["label"]

X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    messages, labels, test_size=0.30, stratify=labels, random_state=42
)

X_val_text, X_test_full, y_val, y_test_full = train_test_split(
    X_temp_text, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# Split between test set and HITL set (to avoid data leakage during HITL simulation)
X_test_text, X_HITL_text, y_test, y_HITL = train_test_split(
    X_test_full, y_test_full, test_size=0.50, stratify=y_test_full, random_state=42
)

# Use noise to create ambiguity and some drift in the test set
test_noisy = X_test_text.apply(lambda msg: character_noise(msg, noise_prob=0.15))
test_noisy_processed = test_noisy.apply(preprocess_text)

HITL_noisy = X_HITL_text.apply(lambda msg: character_noise(msg, noise_prob=0.15))
HITL_noisy_processed = HITL_noisy.apply(preprocess_text)

print("===================================================")

# Base Model - Evaluate on the held-out test set
model_base, vectorizer_base = load_models('v2')
X_test_vectorized_base = vectorizer_base.transform(test_noisy_processed)

before_test_acc, before_test_precision_macro, before_test_recall_macro, before_test_f1_macro, before_test_precision_micro, before_test_recall_micro, before_test_f1_micro, before_test_auroc, before_test_pr_auc = evaluate_model(model_base, X_test_vectorized_base, y_test, dataset_name="Before_HITL", save_conf_matrix=False)
print("HITL Simulation - Before - Test Metrics:")
print(f"HITL Simulation - Before - Test Accuracy: {before_test_acc:.4f}")
print(f"HITL Simulation - Before - Test Precision (Macro): {before_test_precision_macro:.4f}")
print(f"HITL Simulation - Before - Test Recall (Macro): {before_test_recall_macro:.4f}")
print(f"HITL Simulation - Before - Test F1 Score (Macro): {before_test_f1_macro:.4f}")
print(f"HITL Simulation - Before - Test Precision (Micro): {before_test_precision_micro:.4f}")
print(f"HITL Simulation - Before - Test Recall (Micro): {before_test_recall_micro:.4f}")
print(f"HITL Simulation - Before - Test F1 Score (Micro): {before_test_f1_micro:.4f}")
print(f"HITL Simulation - Before - Test AUROC: {before_test_auroc:.4f}")
print(f"HITL Simulation - Before - Test PR-AUC: {before_test_pr_auc:.4f}")

print("===================================================")

# Simulated Active Learning - Use HITL pool (separate from evaluation test set)
X_HITL_vectorized = vectorizer_base.transform(HITL_noisy_processed)
probs = model_base.predict_proba(X_HITL_vectorized)

max_probs = probs.max(axis=1)

ABSTAIN_THRESHOLD = 0.65
APPROVE_THRESHOLD = 0.70
MAX_SAMPLES_FOR_HITL = 25

selected_indices = []
human_labels = []

for i, p in enumerate(max_probs):
    if p < ABSTAIN_THRESHOLD:
        selected_indices.append(i)
        human_labels.append(y_HITL.iloc[i])  # Simulate human providing the correct label from HITL pool
    elif p < APPROVE_THRESHOLD:
        selected_indices.append(i)
        human_labels.append(y_HITL.iloc[i])  # Simulate human approving the model's prediction
    else:
        continue  # Model is confident, no human needed

if(len(selected_indices) > MAX_SAMPLES_FOR_HITL):
    selected_indices = selected_indices[:MAX_SAMPLES_FOR_HITL]
    human_labels = human_labels[:MAX_SAMPLES_FOR_HITL]

num_selected = len(selected_indices)
print(f"Human-labeled samples selected: {num_selected} / {len(X_HITL_text)}")

# Active Learning/HITL Adaptation - Combine original and new human-labeled data + Retraining
# Get the preprocessed text at selected indices (not the vectorized features)
X_new_processed = HITL_noisy_processed.iloc[selected_indices].reset_index(drop=True)
X_train_text_processed = X_train_text.apply(preprocess_text)

y_new = pd.Series(human_labels)

X_train_combined = pd.concat([X_train_text_processed, X_new_processed], ignore_index=True)
y_train_combined = pd.concat([y_train, y_new], ignore_index=True)

vectorizer_HITL = TfidfVectorizer(stop_words='english')
X_train_combined_vectorized = vectorizer_HITL.fit_transform(X_train_combined)

model_HITL, training_time, iterations_used, time_per_iteration, memory_used, size_in_mb, n_parameters, flops = train_model(X_train_combined_vectorized, y_train_combined)
dump(model_HITL, '../models/logistic_regression_model_HITL.joblib')
dump(vectorizer_HITL, '../models/tfidf_vectorizer_HITL.joblib')

print("Active Learning / HITL Model training completed. Metrics:")
print(f"Training completed in {training_time:.4f} milliseconds")
print(f"Iterations used: {iterations_used}")
print(f"Time per iteration: {time_per_iteration * 1000:.4f} milliseconds")
print(f"Memory used during training: {memory_used:.4f} MB")
print(f"Model size on disk: {size_in_mb:.4f} MB")
print(f"Number of parameters: {n_parameters}")
print(f"Estimated total FLOPS: {flops:.2e}")

print("===================================================")

# After Active Learning / HITL - Evaluate on the same held-out test set (NOT the HITL pool)
X_test_vectorized_hitl = vectorizer_HITL.transform(test_noisy_processed)

after_test_acc, after_test_precision_macro, after_test_recall_macro, after_test_f1_macro, after_test_precision_micro, after_test_recall_micro, after_test_f1_micro, after_test_auroc, after_test_pr_auc = evaluate_model(model_HITL, X_test_vectorized_hitl, y_test, dataset_name="After_HITL", save_conf_matrix=False)
print("HITL Simulation - After - Test Metrics:")
print(f"HITL Simulation - After - Test Accuracy: {after_test_acc:.4f}")
print(f"HITL Simulation - After - Test Precision (Macro): {after_test_precision_macro:.4f}")
print(f"HITL Simulation - After - Test Recall (Macro): {after_test_recall_macro:.4f}")
print(f"HITL Simulation - After - Test F1 Score (Macro): {after_test_f1_macro:.4f}")
print(f"HITL Simulation - After - Test Precision (Micro): {after_test_precision_micro:.4f}")
print(f"HITL Simulation - After - Test Recall (Micro): {after_test_recall_micro:.4f}")
print(f"HITL Simulation - After - Test F1 Score (Micro): {after_test_f1_micro:.4f}")
print(f"HITL Simulation - After - Test AUROC: {after_test_auroc:.4f}")
print(f"HITL Simulation - After - Test PR-AUC: {after_test_pr_auc:.4f}")

print("===================================================")

test_noisy_for_metrics = test_noisy_processed.reset_index(drop=True)

latency_single_p50, latency_single_p90, memory_single_mb = measure_single_inference_metrics(test_noisy_for_metrics, model_HITL, vectorizer_HITL)
print("HITL Simulation - Single Inference Metrics:")
print(f"Latency (P50): {latency_single_p50:.4f} ms")
print(f"Latency (P90): {latency_single_p90:.4f} ms")
print(f"Memory Usage: {memory_single_mb:.4f} MB")

print("===================================================")

latency_batch_p50, latency_batch_p90, memory_batch_mb, throughput = measure_batch_inference_metrics(test_noisy_for_metrics, model_HITL, vectorizer_HITL)
print("HITL Simulation - Batch Inference Metrics:")
print(f"Latency (P50): {latency_batch_p50:.4f} ms")
print(f"Latency (P90): {latency_batch_p90:.4f} ms")
print(f"Memory Usage: {memory_batch_mb:.4f} MB")
print(f"Throughput: {throughput:.4f} samples/sec")

print("===================================================")

print(f"Labeling effort: {num_selected / len(X_HITL_text):.2%}")
print(f"Abstained samples in test set: {(max_probs < ABSTAIN_THRESHOLD).sum()}/{len(X_HITL_text)}")
print(f"Approved samples: {((max_probs >= ABSTAIN_THRESHOLD) & (max_probs < APPROVE_THRESHOLD)).sum()}/{len(X_HITL_text)}")