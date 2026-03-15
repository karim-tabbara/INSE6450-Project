import pandas as pd
from inference import load_models
from inference import predict_email
import matplotlib.pyplot as plt


model, vectorizer = load_models('v2')
raw_data_path = '../data/RawData.csv'

df_train = pd.read_csv(raw_data_path)
# To simulate recent-like data, use a subset of the training data
df_recent = df_train.sample(n=200, random_state=42)

# Data drift (message length distribution)
train_lengths = df_train["message"].apply(len)
recent_lengths = df_recent["message"].apply(len)

train_length_mean = train_lengths.mean()
recent_length_mean = recent_lengths.mean()
print(f"Training mean message length: {train_length_mean:.2f}")
print(f"Recent mean message length: {recent_length_mean:.2f}")
print(f"Length shift: {recent_length_mean - train_length_mean:.2f}")

# Model prediction and confidence distribution
predictions = []
confidences = []

for message in df_recent["message"]:
    pred, conf = predict_email(message, model, vectorizer)
    predictions.append(pred)
    confidences.append(conf)

# Print prediction distribution
print("==================================================")
print("Model Prediction Distribution on Recent Data:")
print("==================================================")
prediction_counts = pd.Series(predictions).value_counts().sort_index()
total_predictions = len(predictions)
for pred_class, count in prediction_counts.items():
    if pred_class == 'ABSTAIN':
        percentage = (count / total_predictions) * 100
        print(f"{pred_class}: {count} ({percentage:.2f}%)")
    else:
        print(f"{pred_class}: {count}")
print("==================================================")

# Create combined dashboard with all 3 plots
fig, axes = plt.subplots(3, 1, figsize=(10, 14))

# Subplot 1: Message Length Distribution Drift
axes[0].hist(train_lengths, bins=30, alpha=0.5, label="Train")
axes[0].hist(recent_lengths, bins=30, alpha=0.5, label="Recent")
axes[0].set_xlabel("Message Length")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Message Length Distribution Drift")
axes[0].legend()

# Subplot 2: Confidence Distribution
n, bins, patches = axes[1].hist(confidences, bins=20)
axes[1].set_xlabel("Prediction Confidence")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Model Confidence Distribution on Recent Data")
# Add count labels on top of each bar
for i, patch in enumerate(patches):
    height = patch.get_height()
    if height > 0:  # Only show label if there's a count
        axes[1].text(patch.get_x() + patch.get_width()/2., height,
                     f'{int(height)}',
                     ha='center', va='bottom', fontsize=8)

# Subplot 3: Prediction Distribution
prediction_counts = pd.Series(predictions).value_counts().sort_index()
bars = axes[2].bar(range(len(prediction_counts)), prediction_counts.values)
axes[2].set_xlabel("Predicted Class")
axes[2].set_ylabel("Frequency")
axes[2].set_title("Model Prediction Distribution on Recent Data")
axes[2].set_xticks(range(len(prediction_counts)))
axes[2].set_xticklabels(prediction_counts.index, rotation=45)
# Add count labels on top of each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

plt.tight_layout()
plt.savefig("../outputs/monitoring_dashboard_simulation.png", dpi=300, bbox_inches='tight')
plt.close()
