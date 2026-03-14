import random
import string
import pandas as pd
from model import evaluate_model
from sklearn.model_selection import train_test_split
from inference import load_models
from preprocessing import preprocess_text

random.seed(42)

def token_mask(text, mask_prob=0.15):
    words = text.split()
    masked_tokens = []
    for word in words:
        if random.random() < mask_prob:
            masked_tokens.append("[MASK]")
        else:
            masked_tokens.append(word)
    return ' '.join(masked_tokens)

def character_noise(text, noise_prob=0.1):
    noisy_text = []
    for char in text:
        if random.random() < noise_prob and char.isalpha():
            noisy_char = random.choice(string.ascii_letters)
            noisy_text.append(noisy_char)
        else:
            noisy_text.append(char)
    return ''.join(noisy_text)

def truncate_email(text, truncate_prob=0.5, keep_ratio=0.75):
    words = text.split()
    if random.random() < truncate_prob:
        cutoff = int(len(words) * keep_ratio)
        if len(words) > cutoff:
            return ' '.join(words[:cutoff])
    return text

def ood_input(text, ood_prob=0.3, ood_length=10):
    if random.random() < ood_prob:
        random_tokens = [''.join(random.choices(string.ascii_letters + string.digits, k=5)) for _ in range(ood_length)]
        special_tokens = ["$$$", "###", "!!!", "@@@"]
        if random.random() < 0.5:
            return ' '.join(random_tokens) + ' ' + text + ' ' + random.choice(special_tokens)
        else:
            return text + ' ' + ' '.join(random_tokens) + ' ' + random.choice(special_tokens)
    return text


raw_data_path = '../data/RawData.csv'
df = pd.read_csv(raw_data_path)
messages = df["message"]
labels = df["label"]

# model, vectorizer = load_models('v1')
model, vectorizer = load_models('v2')

X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    messages, labels,
    test_size=0.30, 
    stratify=labels,
    random_state=42
)
    
X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp,
    test_size=0.50, 
    stratify=y_temp, 
    random_state=42
)


X_test_masked = X_test_text.apply(token_mask)
X_test_masked = X_test_masked.apply(preprocess_text)
X_test_masked_final = vectorizer.transform(X_test_masked)
test_acc, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_micro, test_recall_micro, test_f1_micro, test_auroc, test_pr_auc = evaluate_model(model, X_test_masked_final, y_test, dataset_name="Masked_Test")
print("Masked - Test Metrics:")
print(f"Masked - Test Accuracy: {test_acc:.4f}")
print(f"Masked - Test Precision (Macro): {test_precision_macro:.4f}")
print(f"Masked - Test Recall (Macro): {test_recall_macro:.4f}")
print(f"Masked - Test F1 Score (Macro): {test_f1_macro:.4f}")
print(f"Masked - Test Precision (Micro): {test_precision_micro:.4f}")
print(f"Masked - Test Recall (Micro): {test_recall_micro:.4f}")
print(f"Masked - Test F1 Score (Micro): {test_f1_micro:.4f}")
print(f"Masked - Test AUROC: {test_auroc:.4f}")
print(f"Masked - Test PR-AUC: {test_pr_auc:.4f}")

print("===================================================")

X_test_noisy = X_test_text.apply(character_noise)
X_test_noisy = X_test_noisy.apply(preprocess_text)
X_test_noisy_final = vectorizer.transform(X_test_noisy)
test_acc, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_micro, test_recall_micro, test_f1_micro, test_auroc, test_pr_auc = evaluate_model(model, X_test_noisy_final, y_test, dataset_name="Noisy_Test")
print("Noisy - Test Metrics:")
print(f"Noisy - Test Accuracy: {test_acc:.4f}")
print(f"Noisy - Test Precision (Macro): {test_precision_macro:.4f}")
print(f"Noisy - Test Recall (Macro): {test_recall_macro:.4f}")
print(f"Noisy - Test F1 Score (Macro): {test_f1_macro:.4f}")
print(f"Noisy - Test Precision (Micro): {test_precision_micro:.4f}")
print(f"Noisy - Test Recall (Micro): {test_recall_micro:.4f}")
print(f"Noisy - Test F1 Score (Micro): {test_f1_micro:.4f}")
print(f"Noisy - Test AUROC: {test_auroc:.4f}")
print(f"Noisy - Test PR-AUC: {test_pr_auc:.4f}")

print("===================================================")

X_test_truncated = X_test_text.apply(truncate_email)
X_test_truncated = X_test_truncated.apply(preprocess_text)
X_test_truncated_final = vectorizer.transform(X_test_truncated)
test_acc, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_micro, test_recall_micro, test_f1_micro, test_auroc, test_pr_auc = evaluate_model(model, X_test_truncated_final, y_test, dataset_name="Truncated_Test")
print("Truncated - Test Metrics:")
print(f"Truncated - Test Accuracy: {test_acc:.4f}")
print(f"Truncated - Test Precision (Macro): {test_precision_macro:.4f}")
print(f"Truncated - Test Recall (Macro): {test_recall_macro:.4f}")
print(f"Truncated - Test F1 Score (Macro): {test_f1_macro:.4f}")
print(f"Truncated - Test Precision (Micro): {test_precision_micro:.4f}")
print(f"Truncated - Test Recall (Micro): {test_recall_micro:.4f}")
print(f"Truncated - Test F1 Score (Micro): {test_f1_micro:.4f}")
print(f"Truncated - Test AUROC: {test_auroc:.4f}")
print(f"Truncated - Test PR-AUC: {test_pr_auc:.4f}")

print("===================================================")

X_test_ood = X_test_text.apply(ood_input)
X_test_ood = X_test_ood.apply(preprocess_text)
X_test_ood_final = vectorizer.transform(X_test_ood)
test_acc, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_micro, test_recall_micro, test_f1_micro, test_auroc, test_pr_auc = evaluate_model(model, X_test_ood_final, y_test, dataset_name="OOD_Test")
print("OOD - Test Metrics:")
print(f"OOD - Test Accuracy: {test_acc:.4f}")
print(f"OOD - Test Precision (Macro): {test_precision_macro:.4f}")
print(f"OOD - Test Recall (Macro): {test_recall_macro:.4f}")
print(f"OOD - Test F1 Score (Macro): {test_f1_macro:.4f}")
print(f"OOD - Test Precision (Micro): {test_precision_micro:.4f}")
print(f"OOD - Test Recall (Micro): {test_recall_micro:.4f}")
print(f"OOD - Test F1 Score (Micro): {test_f1_micro:.4f}")
print(f"OOD - Test AUROC: {test_auroc:.4f}")
print(f"OOD - Test PR-AUC: {test_pr_auc:.4f}")
