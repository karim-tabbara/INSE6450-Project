import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from inference import load_models, predict_email
from preprocessing import preprocess_text
import pandas as pd
from sklearn.calibration import calibration_curve

# Robustness curve
noise_levels = [0, 5, 10, 15, 20]
accuracies = [0.8867, 0.8200, 0.8200, 0.7600, 0.6333]
f1 = [0.8867, 0.8314, 0.8301, 0.7786, 0.6857]
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(noise_levels, accuracies, marker='o')
plt.xlabel("Character Noise (%)")
plt.ylabel("Accuracy")
plt.title("Robustness Curve: Accuracy vs Adversarial Noise")
plt.subplot(1,2,2)
plt.plot(noise_levels, f1, marker='o')
plt.xlabel("Character Noise (%)")
plt.ylabel("F1 Score (Macro)")
plt.title("Robustness Curve: F1 Score vs Adversarial Noise")
plt.tight_layout()
plt.savefig("../outputs/robustness_curve.png")
plt.close()

# Confidence histogram
model, vectorizer = load_models('v2')
raw_data_path = '../data/RawData.csv'
df = pd.read_csv(raw_data_path)
messages = df["message"]
labels = df["label"]
X_train_text, X_temp_text, y_train, y_temp = train_test_split(messages, labels, test_size=0.30, stratify=labels, random_state=42)
X_val_text, X_test_text, y_val, y_test = train_test_split(X_temp_text, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
X_test = X_test_text.apply(preprocess_text)
X_test_vectorized = vectorizer.transform(X_test)
probs = model.predict_proba(X_test_vectorized)
confidences = probs.max(axis=1)
plt.hist(confidences, bins=20)
plt.xlabel("Prediction Confidence")
plt.ylabel("Frequency")
plt.title("Prediction Confidence Distribution")
plt.savefig("../outputs/confidence_histogram.png")
plt.close()

# Calibration curve
y_pred = model.predict(X_test_vectorized)
is_correct = (y_pred == y_test).astype(int)  # 1 if correct, 0 if wrong
confidences = probs.max(axis=1)

prob_true, prob_pred = calibration_curve(is_correct, confidences, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0,1],[0,1], linestyle='--', label='Perfect Calibration')
plt.xlabel("Predicted Confidence")
plt.ylabel("Actual Accuracy")
plt.title("Reliability Diagram")
plt.legend()
plt.savefig("../outputs/reliability_diagram.png")
plt.close()

# Failure examples extraction
y_pred = model.predict(X_test_vectorized)
probs = model.predict_proba(X_test_vectorized)
confidences = probs.max(axis=1)
results = pd.DataFrame({
    "message": X_test_text,
    "true_label": y_test,
    "predicted_label": y_pred,
    "confidence": confidences
})
failures = results[results["true_label"] != results["predicted_label"]]
failures = failures.sort_values(by="confidence", ascending=False)
failure_examples = failures.head(10)
failure_examples.to_csv("../outputs/failure_examples.csv", index=False)


# Resolves cases
label, confidence = predict_email("Please share your availability for a brief discussion when possible.", model, vectorizer)
print(f"Resolved Case 1 - Predicted Label: {label}, Confidence: {confidence:.4f}")
label, confidence = predict_email("Dear William, Major project timeline changes due to resource reallocation. Review new dates below immediately. Sincerely, William Lester MD.", model, vectorizer)
print(f"Resolved Case 2 - Predicted Label: {label}, Confidence: {confidence:.4f}")
