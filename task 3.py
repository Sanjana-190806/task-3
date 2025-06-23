import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
file_path = r"C:\Users\sanja\Downloads\dataset.csv"
df = pd.read_csv(file_path)

# Show dataset structure
print("Dataset columns:", df.columns.tolist())
print("Dataset head:")
print(df.head())

# Extract relevant columns
source_texts = df['source_text']
plagiarized_texts = df['plagiarized_text']

# Combine all texts for TF-IDF vectorization
all_texts = source_texts.tolist() + plagiarized_texts.tolist()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Split back into source and plagiarized vectors
source_vecs = tfidf_matrix[:len(source_texts)]
plagiarized_vecs = tfidf_matrix[len(source_texts):]

# Calculate cosine similarity between each source-plagiarized pair
similarities = []
for i in range(len(source_texts)):
    sim = cosine_similarity(source_vecs[i], plagiarized_vecs[i])[0][0]
    similarities.append(sim)

# Add similarity scores and prediction to dataframe
df['similarity_score'] = similarities
df['predicted_label'] = df['similarity_score'] > 0.8  # Threshold can be adjusted
df['predicted_label'] = df['predicted_label'].astype(int)  # Convert boolean to 0/1

# Evaluate simple accuracy
accuracy = (df['label'] == df['predicted_label']).mean()
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Show results
print(df[['source_text', 'plagiarized_text', 'similarity_score', 'label', 'predicted_label']])