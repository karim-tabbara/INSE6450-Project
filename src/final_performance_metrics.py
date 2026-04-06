import time
from joblib import dump
from model import evaluate_model, train_model, plot_learning_curve
from preprocessing import load_and_prepare_data, preprocess_text
from inference import measure_single_inference_metrics, measure_batch_inference_metrics, predict_email, load_models
from stress_tests import character_noise
import pandas as pd
from sklearn.model_selection import train_test_split

raw_data_path = '../data/RawData.csv'
df = pd.read_csv(raw_data_path)
messages = df["message"]
labels = df["label"]

model, vectorizer = load_models('v2')
model_CL, vectorizer_CL = load_models('CL')
model_HITL, vectorizer_HITL = load_models('HITL')

# Prepare Test Sets
X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    messages, labels, test_size=0.30, stratify=labels, random_state=42
)
X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

X_test_clean_processed = X_test_text.apply(preprocess_text)
X_test_clean_vectorized = vectorizer.transform(X_test_clean_processed)

X_test_noisy = X_test_text.apply(lambda x: character_noise(x, noise_prob=0.15))
X_test_noisy_processed = X_test_noisy.apply(preprocess_text)
X_test_noisy_vectorized = vectorizer.transform(X_test_noisy_processed)

X_test_CL_vectorized = vectorizer_CL.transform(X_test_noisy_processed)

X_test_HITL_vectorized = vectorizer_HITL.transform(X_test_noisy_processed)

# Run tests
test_acc, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_micro, test_recall_micro, test_f1_micro, test_auroc, test_pr_auc = evaluate_model(model, X_test_clean_vectorized, y_test, dataset_name="Test - Clean", save_conf_matrix=False)
print("Clean Data - Test Metrics:")
print(f"Clean Data - Test Accuracy: {test_acc:.4f}")
print(f"Clean Data - Test Precision (Macro): {test_precision_macro:.4f}")
print(f"Clean Data - Test Recall (Macro): {test_recall_macro:.4f}")
print(f"Clean Data - Test F1 Score (Macro): {test_f1_macro:.4f}")
print(f"Clean Data - Test Precision (Micro): {test_precision_micro:.4f}")
print(f"Clean Data - Test Recall (Micro): {test_recall_micro:.4f}")
print(f"Clean Data - Test F1 Score (Micro): {test_f1_micro:.4f}")
print(f"Clean Data - Test AUROC: {test_auroc:.4f}")
print(f"Clean Data - Test PR-AUC: {test_pr_auc:.4f}")

print("===================================================")

test_acc, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_micro, test_recall_micro, test_f1_micro, test_auroc, test_pr_auc = evaluate_model(model, X_test_noisy_vectorized, y_test, dataset_name="Test - Noisy", save_conf_matrix=False)
print("Noisy Data - Test Metrics:")
print(f"Noisy Data - Test Accuracy: {test_acc:.4f}")
print(f"Noisy Data - Test Precision (Macro): {test_precision_macro:.4f}")
print(f"Noisy Data - Test Recall (Macro): {test_recall_macro:.4f}")
print(f"Noisy Data - Test F1 Score (Macro): {test_f1_macro:.4f}")
print(f"Noisy Data - Test Precision (Micro): {test_precision_micro:.4f}")
print(f"Noisy Data - Test Recall (Micro): {test_recall_micro:.4f}")
print(f"Noisy Data - Test F1 Score (Micro): {test_f1_micro:.4f}")
print(f"Noisy Data - Test AUROC: {test_auroc:.4f}")
print(f"Noisy Data - Test PR-AUC: {test_pr_auc:.4f}")

print("===================================================")

test_acc, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_micro, test_recall_micro, test_f1_micro, test_auroc, test_pr_auc = evaluate_model(model_CL, X_test_CL_vectorized, y_test, dataset_name="Test - CL - Noisy", save_conf_matrix=False)
print("CL - Noisy Data - Test Metrics:")
print(f"CL - Noisy Data - Test Accuracy: {test_acc:.4f}")
print(f"CL - Noisy Data - Test Precision (Macro): {test_precision_macro:.4f}")
print(f"CL - Noisy Data - Test Recall (Macro): {test_recall_macro:.4f}")
print(f"CL - Noisy Data - Test F1 Score (Macro): {test_f1_macro:.4f}")
print(f"CL - Noisy Data - Test Precision (Micro): {test_precision_micro:.4f}")
print(f"CL - Noisy Data - Test Recall (Micro): {test_recall_micro:.4f}")
print(f"CL - Noisy Data - Test F1 Score (Micro): {test_f1_micro:.4f}")
print(f"CL - Noisy Data - Test AUROC: {test_auroc:.4f}")
print(f"CL - Noisy Data - Test PR-AUC: {test_pr_auc:.4f}")

print("===================================================")

test_acc, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_micro, test_recall_micro, test_f1_micro, test_auroc, test_pr_auc = evaluate_model(model_HITL, X_test_HITL_vectorized, y_test, dataset_name="Test - HITL - Noisy", save_conf_matrix=False)
print("HITL - Noisy Data - Test Metrics:")
print(f"HITL - Noisy Data - Test Accuracy: {test_acc:.4f}")
print(f"HITL - Noisy Data - Test Precision (Macro): {test_precision_macro:.4f}")
print(f"HITL - Noisy Data - Test Recall (Macro): {test_recall_macro:.4f}")
print(f"HITL - Noisy Data - Test F1 Score (Macro): {test_f1_macro:.4f}")
print(f"HITL - Noisy Data - Test Precision (Micro): {test_precision_micro:.4f}")
print(f"HITL - Noisy Data - Test Recall (Micro): {test_recall_micro:.4f}")
print(f"HITL - Noisy Data - Test F1 Score (Micro): {test_f1_micro:.4f}")
print(f"HITL - Noisy Data - Test AUROC: {test_auroc:.4f}")
print(f"HITL - Noisy Data - Test PR-AUC: {test_pr_auc:.4f}")