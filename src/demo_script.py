from joblib import dump
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from inference import load_models, predict_email
from preprocessing import preprocess_text
from stress_tests import character_noise
from model import train_model

# Load
model, vectorizer = load_models("v2")
print("[LOAD] Model and vectorizer loaded successfully.")

df = pd.read_csv('../data/RawData.csv')
messages = df["message"]
labels = df["label"]

X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    messages, labels, test_size=0.30, stratify=labels, random_state=42
)

X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

email_to_infer = "I urgently need the updated cost estimate before we can proceed with finalizing the contract terms."
user_label = "Urgent"

# Infer
predicted_class, probability = predict_email(email_to_infer, model, vectorizer)

if predicted_class == "ABSTAIN":
    print(f"\n[INFER] Model abstained from making a prediction on email - \"{email_to_infer}\". Max probability: {probability:.4f}")
    user_label = input("[QUERY HUMAN] Enter the label for this email: ")
elif probability < 0.70:
    print(f"\n[INFER] Predicted class for email - \"{email_to_infer}\": {predicted_class}, Probability: {probability:.4f}")
    user_choice = input("[QUERY HUMAN] Do you validate this label (y/n)? ")
    if user_choice.lower() == 'y':
        user_label = predicted_class
    else:
        user_label = input("Enter the correct label for this email: ")
else:
    print(f"[INFER] Predicted class for email - \"{email_to_infer}\": {predicted_class}, Probability: {probability:.4f}")

# Simulate and Detect Drift
noisy_email = character_noise(email_to_infer, noise_prob=0.15)
print(f"\n[DRIFT] Drift detected - Drifted email: {noisy_email}")

predicted_class_noisy, probability_noisy = predict_email(noisy_email, model, vectorizer)
# Query Human
if predicted_class_noisy == "ABSTAIN":
    print(f"\n[DRIFT - INFER] Model abstained from making a prediction on email - \"{noisy_email}\". Max probability: {probability_noisy:.4f}")
    user_label = input("[QUERY HUMAN] Enter the label for this email: ")
elif probability_noisy < 0.70:
    print(f"\n[DRIFT - INFER] Predicted class for email - \"{noisy_email}\": {predicted_class_noisy}, Probability: {probability_noisy:.4f}")
    user_choice = input("[QUERY HUMAN] Do you validate this label (y/n)? ")
    if user_choice.lower() == 'y':
        user_label = predicted_class_noisy
    else:
        user_label = input("Enter the correct label for this email: ")
else:
    print(f"\n[DRIFT - INFER] Predicted class for email - \"{noisy_email}\": {predicted_class_noisy}, Probability: {probability_noisy:.4f}. No intervention needed.")

# Retrain, Update Model
X_train_text_processed = X_train_text.apply(preprocess_text)
X_noisy_email_processed = pd.Series([preprocess_text(noisy_email)])

X_train_combined = pd.concat([X_train_text_processed, X_noisy_email_processed], ignore_index=True)
y_train_combined = pd.concat([y_train, pd.Series([user_label])], ignore_index=True)

vectorizer_updated = TfidfVectorizer(stop_words='english')
X_train_combined_vectorized = vectorizer_updated.fit_transform(X_train_combined)

print(f"\n[UPDATE] Retraining model with new data point...")
model_updated, training_time, iterations_used, time_per_iteration, memory_used, size_in_mb, n_parameters, flops = train_model(X_train_combined_vectorized, y_train_combined)
print("[UPDATE] Model retrained successfully.")
dump(model_updated, '../models/logistic_regression_model_demo_updated.joblib')
dump(vectorizer_updated, '../models/tfidf_vectorizer_demo_updated.joblib')

# Re-infer on new email and same email to see improvement

predicted_class_noisy, probability_noisy = predict_email(noisy_email, model_updated, vectorizer_updated)
# Query Human
if predicted_class_noisy == "ABSTAIN":
    print(f"\n[DRIFT - INFER] Model abstained from making a prediction on email - \"{noisy_email}\". Max probability: {probability_noisy:.4f}")
    user_label = input("[QUERY HUMAN] Enter the label for this email: ")
elif probability_noisy < 0.70:
    print(f"\n[DRIFT - INFER] Predicted class for email - \"{noisy_email}\": {predicted_class_noisy}, Probability: {probability_noisy:.4f}")
    user_choice = input("[QUERY HUMAN] Do you validate this label (y/n)? ")
    if user_choice.lower() == 'y':
        user_label = predicted_class_noisy
    else:
        user_label = input("Enter the correct label for this email: ")
else:
    print(f"\n[DRIFT - INFER] Predicted class for email - \"{noisy_email}\": {predicted_class_noisy}, Probability: {probability_noisy:.4f}. No intervention needed.")


test_email = "This needs immediate attention - the system is down and affecting all customers."
noisy_email_2 = character_noise(test_email, noise_prob=0.15)

predicted_class_updated, probability_updated = predict_email(noisy_email_2, model_updated, vectorizer_updated)
if predicted_class_updated == "ABSTAIN":
    print(f"\n[UPDATE - INFER] Model abstained from making a prediction on email - \"{noisy_email_2}\". Max probability: {probability_updated:.4f}")
    user_label = input("[QUERY HUMAN] Enter the label for this email: ")
elif probability_updated < 0.70:
    print(f"\n[UPDATE - INFER] Predicted class for email - \"{noisy_email_2}\": {predicted_class_updated}, Probability: {probability_updated:.4f}")
    user_choice = input("[QUERY HUMAN] Do you validate this label (y/n)? ")
    if user_choice.lower() == 'y':
        user_label = predicted_class_updated
    else:
        user_label = input("Enter the correct label for this email: ")
else:
    print(f"\n[UPDATE - INFER] Predicted class for email - \"{noisy_email_2}\": {predicted_class_updated}, Probability: {probability_updated:.4f}. No intervention needed.")