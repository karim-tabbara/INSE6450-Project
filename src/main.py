import time
from joblib import dump
from model import evaluate_model, train_model, plot_learning_curve
from preprocessing import load_and_prepare_data
from inference import measure_single_inference_metrics, measure_batch_inference_metrics, predict_email, load_models
import pandas as pd


raw_data_path = '../data/RawData.csv'
df = pd.read_csv(raw_data_path)
messages = df["message"].tolist()

# Original Model (Milestone 2) - Do not use anymore
# X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, vectorizer = load_and_prepare_data(raw_data_path)
# model, training_time, iterations_used, time_per_iteration, memory_used, size_in_mb, n_parameters, flops = train_model(X_train_final, y_train)
# dump(model, '../models/logistic_regression_model.joblib')
# dump(vectorizer, '../models/tfidf_vectorizer.joblib')

# Updated Model (Milestone 3)
X_train_final, y_train, X_val_final, y_val, X_test_final, y_test, vectorizerV2 = load_and_prepare_data(raw_data_path)
modelV2, training_time, iterations_used, time_per_iteration, memory_used, size_in_mb, n_parameters, flops = train_model(X_train_final, y_train)
dump(modelV2, '../models/logistic_regression_model_v2.joblib')
dump(vectorizerV2, '../models/tfidf_vectorizer_v2.joblib')

print("Model training completed. Metrics:")
print(f"Training completed in {training_time:.4f} milliseconds")
print(f"Iterations used: {iterations_used}")
print(f"Time per iteration: {time_per_iteration * 1000:.4f} milliseconds")
print(f"Memory used during training: {memory_used:.4f} MB")
print(f"Model size on disk: {size_in_mb:.4f} MB")
print(f"Number of parameters: {n_parameters}")
print(f"Estimated total FLOPS: {flops:.2e}")

print("===================================================")

# model, vectorizer = load_models('v1')
model, vectorizer = load_models('v2')

validation_acc, validation_precision_macro, validation_recall_macro, validation_f1_macro, validation_precision_micro, validation_recall_micro, validation_f1_micro, validation_auroc, validation_pr_auc = evaluate_model(model, X_val_final, y_val, dataset_name="Validation")
print("Validation Metrics:")
print(f"Validation Accuracy: {validation_acc:.4f}")
print(f"Validation Precision (Macro): {validation_precision_macro:.4f}")
print(f"Validation Recall (Macro): {validation_recall_macro:.4f}")
print(f"Validation F1 Score (Macro): {validation_f1_macro:.4f}")
print(f"Validation Precision (Micro): {validation_precision_micro:.4f}")
print(f"Validation Recall (Micro): {validation_recall_micro:.4f}")
print(f"Validation F1 Score (Micro): {validation_f1_micro:.4f}")
print(f"Validation AUROC: {validation_auroc:.4f}")
print(f"Validation PR-AUC: {validation_pr_auc:.4f}")

print("===================================================")

test_acc, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_micro, test_recall_micro, test_f1_micro, test_auroc, test_pr_auc = evaluate_model(model, X_test_final, y_test, dataset_name="Test")
print("Test Metrics:")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision (Macro): {test_precision_macro:.4f}")
print(f"Test Recall (Macro): {test_recall_macro:.4f}")
print(f"Test F1 Score (Macro): {test_f1_macro:.4f}")
print(f"Test Precision (Micro): {test_precision_micro:.4f}")
print(f"Test Recall (Micro): {test_recall_micro:.4f}")
print(f"Test F1 Score (Micro): {test_f1_micro:.4f}")
print(f"Test AUROC: {test_auroc:.4f}")
print(f"Test PR-AUC: {test_pr_auc:.4f}")

print("===================================================")

plot_learning_curve(X_train_final, y_train, X_val_final, y_val)
print("Learning curve plotted and saved to ../outputs/learning_curve.png")

print("===================================================")

latency_single_p50, latency_single_p90, memory_single_mb = measure_single_inference_metrics(messages, model, vectorizer)
print("Single Inference Metrics:")
print(f"Latency (P50): {latency_single_p50:.4f} ms")
print(f"Latency (P90): {latency_single_p90:.4f} ms")
print(f"Memory Usage: {memory_single_mb:.4f} MB")

print("===================================================")

latency_batch_p50, latency_batch_p90, memory_batch_mb, throughput = measure_batch_inference_metrics(messages, model, vectorizer)
print("Batch Inference Metrics:")
print(f"Latency (P50): {latency_batch_p50:.4f} ms")
print(f"Latency (P90): {latency_batch_p90:.4f} ms")
print(f"Memory Usage: {memory_batch_mb:.4f} MB")
print(f"Throughput: {throughput:.4f} samples/sec")