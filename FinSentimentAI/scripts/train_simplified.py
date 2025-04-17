import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LABELED_DATA_PATH = os.path.join(DATA_DIR, "labeled_data.csv")
OUTPUT_DIR = os.path.join(MODELS_DIR, "financial-sentiment-model")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters
MODEL_NAME = "distilbert-base-uncased"  # Smaller model that can run on CPU
NUM_LABELS = 3
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

def train_model():
    print("Loading dataset...")
    # Load the labeled data
    df = pd.read_csv(LABELED_DATA_PATH)
    
    # Map sentiment labels to numeric values
    label_map = {"positive": 0, "negative": 1, "neutral": 2}
    df["label"] = df["sentiment"].map(label_map)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])
    
    # Print dataset stats
    print(f"Training examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")
    print("Label distribution:")
    print(train_df["sentiment"].value_counts())
    
    # Load tokenizer and model
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples["news_title"], padding="max_length", truncation=True, max_length=128)
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    
    # Define compute_metrics function for evaluation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Print classification report
        target_names = ["positive", "negative", "neutral"]
        report = classification_report(labels, predictions, target_names=target_names)
        print("\nClassification Report:")
        print(report)
        
        # Calculate accuracy
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Training model...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Model training complete!")
    return OUTPUT_DIR

if __name__ == "__main__":
    train_model()
