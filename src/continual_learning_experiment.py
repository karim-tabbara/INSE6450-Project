from joblib import dump
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from inference import load_models, measure_batch_inference_metrics, measure_single_inference_metrics
from model import evaluate_model, train_model
from preprocessing import preprocess_text
from stress_tests import character_noise
from sklearn.feature_extraction.text import TfidfVectorizer


raw_data_path = '../data/RawData.csv'
df_original = pd.read_csv(raw_data_path)

messages = df_original["message"]
labels = df_original["label"]

# Train/val/test split on ORIGINAL clean data
X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    messages, labels, test_size=0.30, stratify=labels, random_state=42
)

X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# Create noisy versions (simulate drift)
train_noisy = X_train_text.apply(lambda msg: character_noise(msg, noise_prob=0.15))
test_noisy = X_test_text.apply(lambda msg: character_noise(msg, noise_prob=0.17))

X_train_text_processed = X_train_text.apply(preprocess_text)
train_noisy_processed = train_noisy.apply(preprocess_text)
test_noisy_processed = test_noisy.apply(preprocess_text)

print("===================================================")

# Before Continual Learning
model_base, vectorizer_base = load_models('v2')
X_test_base = vectorizer_base.transform(test_noisy_processed)
before_test_acc, before_test_precision_macro, before_test_recall_macro, before_test_f1_macro, before_test_precision_micro, before_test_recall_micro, before_test_f1_micro, before_test_auroc, before_test_pr_auc = evaluate_model(model_base, X_test_base, y_test, dataset_name="Before_CL", save_conf_matrix=False)
print("CL Experiment - Before - Test Metrics:")
print(f"CL Experiment - Before - Test Accuracy: {before_test_acc:.4f}")
print(f"CL Experiment - Before - Test Precision (Macro): {before_test_precision_macro:.4f}")
print(f"CL Experiment - Before - Test Recall (Macro): {before_test_recall_macro:.4f}")
print(f"CL Experiment - Before - Test F1 Score (Macro): {before_test_f1_macro:.4f}")
print(f"CL Experiment - Before - Test Precision (Micro): {before_test_precision_micro:.4f}")
print(f"CL Experiment - Before - Test Recall (Micro): {before_test_recall_micro:.4f}")
print(f"CL Experiment - Before - Test F1 Score (Micro): {before_test_f1_micro:.4f}")
print(f"CL Experiment - Before - Test AUROC: {before_test_auroc:.4f}")
print(f"CL Experiment - Before - Test PR-AUC: {before_test_pr_auc:.4f}")

print("===================================================")

# Continual Learning Adaptation - Combine original and noisy data + Retraining
X_train_combined = pd.concat([X_train_text_processed, train_noisy_processed], ignore_index=True)
y_train_combined = pd.concat([y_train, y_train], ignore_index=True)

vectorizer_CL = TfidfVectorizer(stop_words='english')

X_train_CL = vectorizer_CL.fit_transform(X_train_combined)

model_CL, training_time, iterations_used, time_per_iteration, memory_used, size_in_mb, n_parameters, flops = train_model(X_train_CL, y_train_combined)
dump(model_CL, '../models/logistic_regression_model_CL.joblib')
dump(vectorizer_CL, '../models/tfidf_vectorizer_CL.joblib')

print("Continual Learning Model training completed. Metrics:")
print(f"Training completed in {training_time:.4f} milliseconds")
print(f"Iterations used: {iterations_used}")
print(f"Time per iteration: {time_per_iteration * 1000:.4f} milliseconds")
print(f"Memory used during training: {memory_used:.4f} MB")
print(f"Model size on disk: {size_in_mb:.4f} MB")
print(f"Number of parameters: {n_parameters}")
print(f"Estimated total FLOPS: {flops:.2e}")

print("===================================================")

# After Continual Learning
X_test_CL = vectorizer_CL.transform(test_noisy_processed)

after_test_acc, after_test_precision_macro, after_test_recall_macro, after_test_f1_macro, after_test_precision_micro, after_test_recall_micro, after_test_f1_micro, after_test_auroc, after_test_pr_auc = evaluate_model(model_CL, X_test_CL, y_test, dataset_name="After_CL", save_conf_matrix=False)
print("CL Experiment - After - Test Metrics:")
print(f"CL Experiment - After - Test Accuracy: {after_test_acc:.4f}")
print(f"CL Experiment - After - Test Precision (Macro): {after_test_precision_macro:.4f}")
print(f"CL Experiment - After - Test Recall (Macro): {after_test_recall_macro:.4f}")
print(f"CL Experiment - After - Test F1 Score (Macro): {after_test_f1_macro:.4f}")
print(f"CL Experiment - After - Test Precision (Micro): {after_test_precision_micro:.4f}")
print(f"CL Experiment - After - Test Recall (Micro): {after_test_recall_micro:.4f}")
print(f"CL Experiment - After - Test F1 Score (Micro): {after_test_f1_micro:.4f}")
print(f"CL Experiment - After - Test AUROC: {after_test_auroc:.4f}")
print(f"CL Experiment - After - Test PR-AUC: {after_test_pr_auc:.4f}")

print("===================================================")

test_noisy_for_metrics = test_noisy_processed.reset_index(drop=True)

latency_single_p50, latency_single_p90, memory_single_mb = measure_single_inference_metrics(test_noisy_for_metrics, model_CL, vectorizer_CL)
print("Continual Learning Experiment - Single Inference Metrics:")
print(f"Latency (P50): {latency_single_p50:.4f} ms")
print(f"Latency (P90): {latency_single_p90:.4f} ms")
print(f"Memory Usage: {memory_single_mb:.4f} MB")

print("===================================================")

latency_batch_p50, latency_batch_p90, memory_batch_mb, throughput = measure_batch_inference_metrics(test_noisy_for_metrics, model_CL, vectorizer_CL)
print("Continual Learning Experiment - Batch Inference Metrics:")
print(f"Latency (P50): {latency_batch_p50:.4f} ms")
print(f"Latency (P90): {latency_batch_p90:.4f} ms")
print(f"Memory Usage: {memory_batch_mb:.4f} MB")
print(f"Throughput: {throughput:.4f} samples/sec")

print("===================================================")

noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]

base_model_accuracies = []
base_model_f1 = []
cl_model_accuracies = []
cl_model_f1 = []

for noise in noise_levels:
    noisy_test_messages = X_test_text.apply(lambda msg: character_noise(msg, noise_prob=noise))
    noisy_test_messages = noisy_test_messages.apply(preprocess_text)
    
    noisy_test_features_base = vectorizer_base.transform(noisy_test_messages)
    acc_base, _, _, f1_base, _, _, _, _, _ = evaluate_model(model_base, noisy_test_features_base, y_test, dataset_name=f"CL_Experiment_Noise_{noise:.2f}_Base", save_conf_matrix=False)
    base_model_accuracies.append(acc_base)
    base_model_f1.append(f1_base)

    noisy_test_features_cl = vectorizer_CL.transform(noisy_test_messages)
    acc_cl, _, _, f1_cl, _, _, _, _, _ = evaluate_model(model_CL, noisy_test_features_cl, y_test, dataset_name=f"CL_Experiment_Noise_{noise:.2f}_CL", save_conf_matrix=False)
    cl_model_accuracies.append(acc_cl)
    cl_model_f1.append(f1_cl)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot - Accuracy
ax1.plot([n*100 for n in noise_levels], base_model_accuracies, marker='o', label='Base Model Accuracy')
ax1.plot([n*100 for n in noise_levels], cl_model_accuracies, marker='o', label='CL Model Accuracy')
ax1.set_xlabel('Noise Level (%)')
ax1.set_ylabel('Test Accuracy')
ax1.set_title('Test Accuracy vs Noise Level')
ax1.set_ylim([0.0, 1.0])
ax1.set_yticks(np.arange(0.0, 1.05, 0.05))
ax1.legend()

# Right subplot - F1 Score
ax2.plot([n*100 for n in noise_levels], base_model_f1, marker='o', label='Base Model F1 Score')
ax2.plot([n*100 for n in noise_levels], cl_model_f1, marker='o', label='CL Model F1 Score')
ax2.set_xlabel('Noise Level (%)')
ax2.set_ylabel('Test F1 Score')
ax2.set_title('Test F1 Score vs Noise Level')
ax2.set_ylim([0.0, 1.0])
ax2.set_yticks(np.arange(0.0, 1.05, 0.05))
ax2.legend()

plt.tight_layout()
plt.savefig('../outputs/CL_Experiment_Accuracy_and_F1_vs_Noise.png')
plt.close('all')