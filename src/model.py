import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
import psutil


def train_model(X_train, y_train):
    # model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    model = LogisticRegression(max_iter=1000, multi_class="multinomial", C=5.0)

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / (1024 * 1024)  # in MB
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Measure memory usage after training
    memory_after = process.memory_info().rss / (1024 * 1024)  # in MB
    memory_used = memory_after - memory_before

    iterations_used = model.n_iter_[0]

    # Calculate time per iteration and model size
    time_per_iteration = training_time / iterations_used
    size_in_mb = (model.coef_.nbytes + model.intercept_.nbytes) / (1024 * 1024)
    
    # Calculate number of parameters
    n_coefficients = model.coef_.size  # Total number of weights
    n_intercepts = model.intercept_.size  # Total number of biases
    n_parameters = n_coefficients + n_intercepts

    # Calcuate FLOPS
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_classes = len(model.classes_)
    flops = (2 * n_samples * n_features * n_classes * iterations_used)/1e9  # Convert to GFLOPS
    
    return model, training_time, iterations_used, time_per_iteration, memory_used, size_in_mb, n_parameters, flops


# Evaluate the model on validation or test set
def evaluate_model(model, X_val, y_val, dataset_name="Validation"):
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    
    acc = accuracy_score(y_val, y_pred)

    precision_macro = precision_score(y_val, y_pred, average="macro")
    recall_macro = recall_score(y_val, y_pred, average="macro")
    f1_macro = f1_score(y_val, y_pred, average="macro")

    precision_micro = precision_score(y_val, y_pred, average="micro")
    recall_micro = recall_score(y_val, y_pred, average="micro")
    f1_micro = f1_score(y_val, y_pred, average="micro")
    
    auroc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='macro')
    y_val_bin = label_binarize(y_val, classes=model.classes_)
    pr_auc = average_precision_score(y_val_bin, y_pred_proba, average='macro')

    labels = model.classes_
    conf_matrix = confusion_matrix(y_val, y_pred)
    save_confusion_matrix(conf_matrix, labels=labels, output_path=f"../outputs/confusion_matrix_{dataset_name}.png")

    return acc, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro, auroc, pr_auc


def save_confusion_matrix(conf_matrix, labels, output_path="../outputs/confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()