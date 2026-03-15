import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import load_and_prepare_data, preprocess_text
from stress_tests import character_noise
from model import train_model, evaluate_model
from joblib import dump
from inference import measure_single_inference_metrics, measure_batch_inference_metrics

raw_data_path = '../data/RawData.csv'
df = pd.read_csv(raw_data_path)

messages = df["message"]
labels = df["label"]

# Drifted dataset
noisy_messages = messages.apply(character_noise)

# Save noisy data to CSV
noisy_df = pd.DataFrame({
    "message": noisy_messages,
    "label": labels
})
noisy_df.to_csv("../outputs/NoisyData.csv", index=False)

noisy_data_path = '../outputs/NoisyData.csv'
X_noisy_train_final, y_noisy_train, X_noisy_val_final, y_noisy_val, X_noisy_test_final, y_noisy_test, vectorizer_noisy = load_and_prepare_data(noisy_data_path)
model_noisy, training_time, iterations_used, time_per_iteration, memory_used, size_in_mb, n_parameters, flops = train_model(X_noisy_train_final, y_noisy_train)
dump(model_noisy, '../models/logistic_regression_model_noisy.joblib')
dump(vectorizer_noisy, '../models/tfidf_vectorizer_noisy.joblib')

print("===================================================")

print("Noisy Model training completed. Metrics:")
print(f"Training completed in {training_time:.4f} milliseconds")
print(f"Iterations used: {iterations_used}")
print(f"Time per iteration: {time_per_iteration * 1000:.4f} milliseconds")
print(f"Memory used during training: {memory_used:.4f} MB")
print(f"Model size on disk: {size_in_mb:.4f} MB")
print(f"Number of parameters: {n_parameters}")
print(f"Estimated total FLOPS: {flops:.2e}")

print("===================================================")

X_train_text, X_temp_text, y_train, y_temp = train_test_split(noisy_messages, labels, test_size=0.30, stratify=labels, random_state=42)
    
X_val_text, X_test_text, y_val, y_test = train_test_split(X_temp_text, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

X_test_noisy = X_test_text.apply(character_noise)
X_test_noisy = X_test_noisy.apply(preprocess_text)
X_test_noisy_final = vectorizer_noisy.transform(X_test_noisy)
test_acc, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_micro, test_recall_micro, test_f1_micro, test_auroc, test_pr_auc = evaluate_model(model_noisy, X_test_noisy_final, y_test, dataset_name="Noisy_Adaptation_Experiment")
print("Noisy Adaptation Experiment - Test Metrics:")
print(f"Noisy Adaptation Experiment - Test Accuracy: {test_acc:.4f}")
print(f"Noisy Adaptation Experiment - Test Precision (Macro): {test_precision_macro:.4f}")
print(f"Noisy Adaptation Experiment - Test Recall (Macro): {test_recall_macro:.4f}")
print(f"Noisy Adaptation Experiment - Test F1 Score (Macro): {test_f1_macro:.4f}")
print(f"Noisy Adaptation Experiment - Test Precision (Micro): {test_precision_micro:.4f}")
print(f"Noisy Adaptation Experiment - Test Recall (Micro): {test_recall_micro:.4f}")
print(f"Noisy Adaptation Experiment - Test F1 Score (Micro): {test_f1_micro:.4f}")
print(f"Noisy Adaptation Experiment - Test AUROC: {test_auroc:.4f}")
print(f"Noisy Adaptation Experiment - Test PR-AUC: {test_pr_auc:.4f}")


print("===================================================")

# noisy_messages = [character_noise(msg) for msg in messages]
latency_single_p50, latency_single_p90, memory_single_mb = measure_single_inference_metrics(noisy_messages, model_noisy, vectorizer_noisy)
# latency_single_p50, latency_single_p90, memory_single_mb = measure_single_inference_metrics(noisy_messages, model, vectorizer)
print("Noisy Adaptation Experiment - Single Inference Metrics:")
print(f"Latency (P50): {latency_single_p50:.4f} ms")
print(f"Latency (P90): {latency_single_p90:.4f} ms")
print(f"Memory Usage: {memory_single_mb:.4f} MB")

print("===================================================")

latency_batch_p50, latency_batch_p90, memory_batch_mb, throughput = measure_batch_inference_metrics(noisy_messages, model_noisy, vectorizer_noisy)
# latency_batch_p50, latency_batch_p90, memory_batch_mb, throughput = measure_batch_inference_metrics(noisy_messages, model, vectorizer)
print("Noisy Adaptation Experiment - Batch Inference Metrics:")
print(f"Latency (P50): {latency_batch_p50:.4f} ms")
print(f"Latency (P90): {latency_batch_p90:.4f} ms")
print(f"Memory Usage: {memory_batch_mb:.4f} MB")
print(f"Throughput: {throughput:.4f} samples/sec")