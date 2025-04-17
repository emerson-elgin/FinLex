import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LABELED_DATA_PATH = os.path.join(DATA_DIR, "labeled_data.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "financial_sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    return ' '.join(tokens)

def train_model():
    print("Loading dataset...")
    # Load the labeled data
    df = pd.read_csv(LABELED_DATA_PATH)
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['news_title'].apply(preprocess_text)
    
    # Map sentiment labels to numeric values
    label_map = {"positive": 0, "negative": 1, "neutral": 2}
    label_reverse_map = {0: "positive", 1: "negative", 2: "neutral"}
    df["label"] = df["sentiment"].map(label_map)
    
    # Split into train and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['sentiment']
    )
    
    # Print dataset stats
    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples: {len(X_test)}")
    print("Label distribution:")
    print(df["sentiment"].value_counts())
    
    # Create TF-IDF vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train logistic regression model
    print("Training model...")
    model = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
    model.fit(X_train_vec, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["positive", "negative", "neutral"]))
    
    # Save the model and vectorizer
    print(f"Saving model to {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Test prediction
    test_texts = [
        "The company reported a 20% increase in quarterly profits, exceeding analyst expectations.",
        "The stock plummeted after the company announced significant losses due to regulatory fines.",
        "The market remained stable despite fluctuations in international trade relationships."
    ]
    
    print("\nTesting predictions on sample texts:")
    test_processed = [preprocess_text(text) for text in test_texts]
    test_vectors = vectorizer.transform(test_processed)
    test_predictions = model.predict(test_vectors)
    
    for i, text in enumerate(test_texts):
        sentiment = label_reverse_map[test_predictions[i]]
        print(f"Text: {text}")
        print(f"Predicted sentiment: {sentiment.upper()}\n")
    
    print("✅ Model training complete!")
    return MODEL_PATH, VECTORIZER_PATH

if __name__ == "__main__":
    train_model()
