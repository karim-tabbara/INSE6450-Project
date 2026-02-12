import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import string

df = pd.read_csv('../data/RawData.csv')
print(f"Raw Data shape: {df.shape}")
print(f"Raw Data Label distribution:\n{df.label.value_counts()}")

# OUTPUT ARTIFACT 1: Label Distribution
label_distribution_count = df['label'].value_counts()
bars = plt.bar(label_distribution_count.index, label_distribution_count.values)
plt.xlabel("Labels")
plt.ylabel("Count")
plt.title("Label Distribution")
# Add count labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., 
             height,
             f'{int(height)}',
             ha='center',
             va='bottom')
# Save the figure as an output artifact
plt.savefig('../outputs/labels_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# DATA PROCESSING
# Process the data in RawData.csv and save the processed data as ProcessedData.csv

# Before processing, need to extract a few features
    # Character count
    # Word count
    # "?" count
    # "!" count
unprocessed_text = df["message"]
character_count = unprocessed_text.str.len()
word_count = unprocessed_text.str.split().apply(len)
question_count = unprocessed_text.str.count(r"\?")
exclamation_count = unprocessed_text.str.count("!")

numeric_features = np.column_stack([
    character_count,
    word_count,
    question_count,
    exclamation_count
])
print("=================================================")
print(f"Numeric features shape: {numeric_features.shape}")
print(numeric_features[:3])

# Processing activities: 
    # - lower casing
    # - removing punctuation
    # - removing stop words (done by TfidfVectorizer later)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Apply preprocessing
processed_text = unprocessed_text.apply(preprocess_text)

# Save processed data to CSV with same format as RawData.csv (message, label)
processed_df = pd.DataFrame({
    'message': processed_text,
    'label': df['label']
})
processed_df.to_csv('../outputs/ProcessedData.csv', index=False)
print("=================================================")
print("Processed data saved to ../outputs/ProcessedData.csv")

print(f"Processed Data shape: {processed_df.shape}")
print(f"Processed Data Label distribution:\n{processed_df.label.value_counts()}")

# FEATURE ENGINEERING
# Use TF-IDF to extract features from the text data after processing it
# Concatenate the TF-IDF features with the numeric features extracted above
vectorizer = TfidfVectorizer(stop_words='english')

X_tfidf = vectorizer.fit_transform(processed_df['message'])

print("=================================================")
print(f"TF-IDF features shape: {X_tfidf.shape}")

scaler = StandardScaler(with_mean=False)
X_numeric_scaled = scaler.fit_transform(numeric_features)

X_final = hstack([X_tfidf, X_numeric_scaled])

print("=================================================")
print(f"Final feature matrix shape: {X_final.shape}")