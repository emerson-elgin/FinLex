import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "financial_sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

class FinancialSentimentPredictor:
    def __init__(self, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
        """Initialize the predictor with the trained model and vectorizer"""
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.label_map = {0: "positive", 1: "negative", 2: "neutral"}
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print("Model and vectorizer loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def preprocess_text(self, text):
        """Preprocess text for prediction"""
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and non-alphabetic tokens
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        
        return ' '.join(tokens)
    
    def predict(self, text, include_probabilities=False):
        """
        Predict sentiment for a given text
        
        Args:
            text (str): The financial text to analyze
            include_probabilities (bool): Whether to include class probabilities
            
        Returns:
            dict: Prediction results
        """
        if not self.model or not self.vectorizer:
            return {"error": "Model not loaded"}
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize text
        text_vector = self.vectorizer.transform([processed_text])
        
        # Predict
        sentiment_id = self.model.predict(text_vector)[0]
        sentiment = self.label_map[sentiment_id]
        
        # Create response
        result = {
            "text": text,
            "sentiment": sentiment
        }
        
        # Add probabilities if requested
        if include_probabilities:
            probabilities = self.model.predict_proba(text_vector)[0]
            result["probabilities"] = {
                "positive": float(probabilities[0]),
                "negative": float(probabilities[1]),
                "neutral": float(probabilities[2])
            }
            
            # Calculate confidence (highest probability)
            result["confidence"] = float(max(probabilities))
        
        return result

# Function for simple testing
def test_predictor():
    predictor = FinancialSentimentPredictor()
    
    test_texts = [
        "The company reported a 20% increase in quarterly profits, exceeding analyst expectations.",
        "The stock plummeted after the company announced significant losses due to regulatory fines.",
        "The market remained stable despite fluctuations in international trade relationships.",
        "Investors are concerned about the company's debt levels and its ability to finance future growth."
    ]
    
    print("\n=== Financial Sentiment Analysis Test ===")
    for text in test_texts:
        result = predictor.predict(text, include_probabilities=True)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  - {label.capitalize()}: {prob:.4f}")
    
    return predictor

if __name__ == "__main__":
    predictor = test_predictor()
