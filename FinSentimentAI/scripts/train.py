import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Define paths
DATA_PATH = "../data/labeled_data.csv"
MODEL_SAVE_PATH = "../models/finbert_model"

# Load preprocessed dataset
df = pd.read_csv(DATA_PATH)

# Map sentiment labels to numeric values
label_map = {"positive": 0, "negative": 1, "neutral": 2}
df["sentiment"] = df["sentiment"].map(label_map)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")


# Define Dataset Class
class FinancialSentimentDataset(Dataset):
    def __init__(self, data):
        self.texts = data["cleaned_text"].tolist()
        self.labels = data["sentiment"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in encoding.items()}, torch.tensor(self.labels[idx])


# Create train dataset
dataset = FinancialSentimentDataset(df)

# Load FinBERT model
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save trained model
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(f"✅ Model training complete! Model saved to {MODEL_SAVE_PATH}")
