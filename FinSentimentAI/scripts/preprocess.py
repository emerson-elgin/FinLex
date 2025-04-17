import pandas as pd
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Define paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
RAW_NEWS_PATH = os.path.join(DATA_DIR, "raw_news.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_news.csv")

# Load dataset with error handling
try:
    df = pd.read_csv(RAW_NEWS_PATH, encoding="ISO-8859-1")  # Handle encoding issues
except UnicodeDecodeError:
    df = pd.read_csv(RAW_NEWS_PATH, encoding="utf-8")

# Fill missing values
df["news_title"] = df["news_title"].fillna("")

# Cleaning function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()
    tokens = word_tokenize(text) if isinstance(text, str) else []
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply text cleaning
df["cleaned_text"] = df["news_title"].apply(clean_text)

# Save cleaned data
df.to_csv(CLEANED_DATA_PATH, index=False)

print(f"✅ Text Preprocessing Complete! Cleaned data saved to {CLEANED_DATA_PATH}")
