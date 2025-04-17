import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

# Define paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
LABELED_DATA_PATH = os.path.join(DATA_DIR, "labeled_data.csv")
LLAMA_TRAIN_PATH = os.path.join(DATA_DIR, "llama_train.json")
LLAMA_EVAL_PATH = os.path.join(DATA_DIR, "llama_eval.json")

# Define sentiment mapping for Llama format
SENTIMENT_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral"
}

def create_prompt(text, sentiment=None):
    """
    Create a properly formatted prompt for Llama 2.7 fine-tuning
    """
    base_prompt = f"""<s>[INST] Analyze the sentiment of the following financial text:

{text}

Return only one of these sentiment labels: positive, negative, or neutral. [/INST]"""
    
    if sentiment:
        return base_prompt + f" {SENTIMENT_MAP[sentiment]}</s>"
    return base_prompt

def prepare_llama_dataset():
    """
    Convert financial sentiment dataset to Llama 2.7 fine-tuning format
    """
    print("Loading and preparing dataset for Llama 2.7 fine-tuning...")
    
    try:
        # Load dataset
        df = pd.read_csv(LABELED_DATA_PATH)
        
        # Ensure required columns exist
        assert "sentiment" in df.columns, "Missing 'sentiment' column in dataset"
        assert "news_title" in df.columns, "Missing 'news_title' column in dataset"
        
        # Split into train/evaluation sets (80/20)
        train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])
        
        # Convert to Llama instruction format
        train_data = []
        for _, row in train_df.iterrows():
            train_data.append({
                "text": create_prompt(row["news_title"], row["sentiment"])
            })
        
        eval_data = []
        for _, row in eval_df.iterrows():
            eval_data.append({
                "text": create_prompt(row["news_title"], row["sentiment"])
            })
        
        # Save as JSON files
        with open(LLAMA_TRAIN_PATH, "w") as f:
            json.dump(train_data, f, indent=2)
        
        with open(LLAMA_EVAL_PATH, "w") as f:
            json.dump(eval_data, f, indent=2)
        
        print(f"✅ Successfully prepared Llama 2.7 datasets:")
        print(f"  - Training data: {len(train_data)} examples saved to {LLAMA_TRAIN_PATH}")
        print(f"  - Evaluation data: {len(eval_data)} examples saved to {LLAMA_EVAL_PATH}")
        
    except Exception as e:
        print(f"❌ Error preparing dataset: {str(e)}")

if __name__ == "__main__":
    prepare_llama_dataset()
