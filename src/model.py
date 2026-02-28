import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, multi_class="multinomial", C=5.0)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print("Iterations used: ", model.n_iter_)
    
    return model, training_time


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