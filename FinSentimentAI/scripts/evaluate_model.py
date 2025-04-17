import os
import json
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LLAMA_EVAL_PATH = os.path.join(DATA_DIR, "llama_eval.json")
MODEL_PATH = os.path.join(MODELS_DIR, "llama-2-7b-finsentiment")
RESULTS_PATH = os.path.join(BASE_DIR, "evaluation_results.json")

# Base model ID
BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"

def extract_sentiment_label(generated_text):
    """
    Extract sentiment label from model output
    """
    # Extract the last word which should be the sentiment label
    last_word = generated_text.split()[-1].lower().strip('.,;:!?')
    
    # Check if last word is one of the expected sentiment labels
    if last_word in ["positive", "negative", "neutral"]:
        return last_word
    
    # If no sentiment is found in the last word, check for sentiment keywords in the text
    if "positive" in generated_text.lower():
        return "positive"
    elif "negative" in generated_text.lower():
        return "negative"
    elif "neutral" in generated_text.lower():
        return "neutral"
    
    # Default fallback
    return "neutral"

def load_test_data():
    """
    Load the test dataset
    """
    with open(LLAMA_EVAL_PATH, 'r') as f:
        data = json.load(f)
    
    texts = []
    true_labels = []
    
    for item in data:
        # Extract text from instruction format
        parts = item["text"].split("[/INST]")
        
        if len(parts) >= 2:
            instruction = parts[0].split("Analyze the sentiment of the following financial text:")[1].strip()
            target = parts[1].strip()
            
            texts.append(instruction)
            true_labels.append(target)
    
    return texts, true_labels

def evaluate_model():
    """
    Evaluate the fine-tuned model on the test dataset
    """
    print("Loading test data...")
    test_texts, true_labels = load_test_data()
    print(f"Loaded {len(test_texts)} test examples")
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Load the model - check if we have a merged model or need to load adapter
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Loaded merged model")
    except Exception as e:
        print(f"Loading base model with adapter: {str(e)}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    
    print("Generating predictions...")
    predictions = []
    
    for i, text in enumerate(test_texts):
        prompt = f"<s>[INST] Analyze the sentiment of the following financial text:\n\n{text}\n\nReturn only one of these sentiment labels: positive, negative, or neutral. [/INST]"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = extract_sentiment_label(generated_text)
        predictions.append(prediction)
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(test_texts)} examples")
    
    # Convert true labels to the same format as predictions
    processed_true_labels = [extract_sentiment_label(label) for label in true_labels]
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    accuracy = accuracy_score(processed_true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        processed_true_labels, predictions, average='weighted'
    )
    
    # Calculate class-wise metrics
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        processed_true_labels, predictions, average=None,
        labels=['positive', 'negative', 'neutral']
    )
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(
        processed_true_labels, predictions, 
        labels=['positive', 'negative', 'neutral']
    )
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Positive', 'Negative', 'Neutral'],
        yticklabels=['Positive', 'Negative', 'Neutral']
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Financial Sentiment Analysis')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
    
    # Create results dictionary
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "class_metrics": {
            "positive": {
                "precision": float(class_precision[0]),
                "recall": float(class_recall[0]),
                "f1": float(class_f1[0])
            },
            "negative": {
                "precision": float(class_precision[1]),
                "recall": float(class_recall[1]),
                "f1": float(class_f1[1])
            },
            "neutral": {
                "precision": float(class_precision[2]),
                "recall": float(class_recall[2]),
                "f1": float(class_f1[2])
            }
        },
        "confusion_matrix": conf_matrix.tolist()
    }
    
    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClass-wise metrics:")
    print(f"Positive - Precision: {class_precision[0]:.4f}, Recall: {class_recall[0]:.4f}, F1: {class_f1[0]:.4f}")
    print(f"Negative - Precision: {class_precision[1]:.4f}, Recall: {class_recall[1]:.4f}, F1: {class_f1[1]:.4f}")
    print(f"Neutral - Precision: {class_precision[2]:.4f}, Recall: {class_recall[2]:.4f}, F1: {class_f1[2]:.4f}")
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Confusion matrix plot saved to {os.path.join(BASE_DIR, 'confusion_matrix.png')}")

if __name__ == "__main__":
    evaluate_model()
