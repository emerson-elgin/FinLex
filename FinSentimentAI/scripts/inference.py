import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "llama-2-7b-finsentiment")

# Base model ID
BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"

class FinancialSentimentAnalyzer:
    def __init__(self, model_path=MODEL_PATH, base_model_id=BASE_MODEL_ID):
        """
        Initialize the financial sentiment analyzer with a fine-tuned Llama 2.7 model
        """
        self.model_path = model_path
        self.base_model_id = base_model_id
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.load_model()
    
    def load_model(self):
        """
        Load the fine-tuned model and tokenizer
        """
        logger.info(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        try:
            # First try to load as a merged model
            logger.info(f"Attempting to load merged model from {self.model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            logger.info("Successfully loaded merged model")
        except Exception as e:
            # If that fails, load the base model and adapter separately
            logger.info(f"Loading base model with adapter: {str(e)}")
            logger.info(f"Loading base model from {self.base_model_id}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                device_map="auto",
                torch_dtype=torch.float16
            )
            logger.info(f"Loading adapter from {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            logger.info("Successfully loaded base model with adapter")
    
    def analyze_sentiment(self, text, include_reasoning=False):
        """
        Analyze the sentiment of the provided financial text
        
        Args:
            text (str): Financial text to analyze
            include_reasoning (bool): Whether to include the model's reasoning
            
        Returns:
            dict: Sentiment analysis results
        """
        # Prepare the prompt
        if include_reasoning:
            prompt = f"<s>[INST] Analyze the sentiment of the following financial text and explain your reasoning:\n\n{text}\n\nFirst provide your reasoning, then on a new line return exactly one of these sentiment labels: positive, negative, or neutral. [/INST]"
        else:
            prompt = f"<s>[INST] Analyze the sentiment of the following financial text:\n\n{text}\n\nReturn only one of these sentiment labels: positive, negative, or neutral. [/INST]"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100 if include_reasoning else 20,
                temperature=0.1,
                do_sample=False,
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract sentiment and reasoning
        result = {}
        
        if include_reasoning:
            # Split reasoning and sentiment label
            parts = generated_text.split(prompt)[-1].strip().split("\n")
            
            if len(parts) > 1:
                # Last line should contain the sentiment
                sentiment = parts[-1].lower().strip()
                reasoning = "\n".join(parts[:-1]).strip()
            else:
                # Handle case where no clear split between reasoning and sentiment
                sentiment = self._extract_sentiment_label(generated_text)
                reasoning = parts[0].strip()
                
            result = {
                "sentiment": sentiment,
                "reasoning": reasoning,
                "full_text": generated_text
            }
        else:
            # Just extract the sentiment label
            sentiment = self._extract_sentiment_label(generated_text)
            result = {
                "sentiment": sentiment,
                "full_text": generated_text
            }
        
        return result
    
    def _extract_sentiment_label(self, text):
        """
        Extract sentiment label from model output
        """
        # Extract the last word which should be the sentiment label
        last_word = text.split()[-1].lower().strip('.,;:!?')
        
        # Check if last word is one of the expected sentiment labels
        if last_word in ["positive", "negative", "neutral"]:
            return last_word
        
        # If no sentiment is found in the last word, check for sentiment keywords in the text
        if "positive" in text.lower():
            return "positive"
        elif "negative" in text.lower():
            return "negative"
        elif "neutral" in text.lower():
            return "neutral"
        
        # Default fallback
        return "neutral"

# Function for simple command-line usage
def analyze_text(text):
    analyzer = FinancialSentimentAnalyzer()
    result = analyzer.analyze_sentiment(text, include_reasoning=True)
    
    print("\n=== Financial Sentiment Analysis ===")
    print(f"Text: {text}")
    print(f"Sentiment: {result['sentiment'].upper()}")
    print(f"Reasoning: {result['reasoning']}")
    
    return result

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Financial Sentiment Analysis with Llama 2.7")
    parser.add_argument("--text", type=str, help="Financial text to analyze")
    parser.add_argument("--file", type=str, help="Path to file containing financial text")
    
    args = parser.parse_args()
    
    if args.text:
        analyze_text(args.text)
    elif args.file:
        with open(args.file, 'r') as f:
            text = f.read().strip()
        analyze_text(text)
    else:
        # Interactive mode
        print("=== Financial Sentiment Analysis with Llama 2.7 ===")
        print("Enter financial text to analyze (type 'exit' to quit):")
        
        while True:
            text = input("\nText: ")
            if text.lower() in ["exit", "quit", "q"]:
                break
                
            if text.strip():
                analyze_text(text)
