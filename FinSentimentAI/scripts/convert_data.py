import pandas as pd
import os

# Define paths
DATA_DIR = r"C:\Users\ASUS\FinLex\FinSentimentAI\data"
ALL_DATA_PATH = os.path.join(DATA_DIR, "all-data.csv")
RAW_NEWS_PATH = os.path.join(DATA_DIR, "raw_news.csv")
LABELED_DATA_PATH = os.path.join(DATA_DIR, "labeled_data.csv")

# Check if all-data.csv exists
if not os.path.exists(ALL_DATA_PATH):
    print("❌ Error: 'all-data.csv' not found in the data directory. Please add it first.")
    exit()

# Load the dataset with encoding handling
try:
    df = pd.read_csv(ALL_DATA_PATH, encoding="utf-8")  # Try UTF-8 first
except UnicodeDecodeError:
    print("⚠️ UTF-8 failed. Trying ISO-8859-1 encoding...")
    df = pd.read_csv(ALL_DATA_PATH, encoding="ISO-8859-1")  # Alternative encoding

# Ensure correct column names
df.columns = ["sentiment", "news_title"]

# Save raw news titles (only text)
df[["news_title"]].to_csv(RAW_NEWS_PATH, index=False, encoding="utf-8")

# Save labeled dataset (text + sentiment)
df.to_csv(LABELED_DATA_PATH, index=False, encoding="utf-8")

print(f"✅ Data conversion complete! Files saved in {DATA_DIR}")
