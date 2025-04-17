from flask import Flask, render_template, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Download required NLTK data
nltk.download('vader_lexicon')

app = Flask(__name__)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Financial keywords for enhanced analysis
FINANCIAL_KEYWORDS = {
    'positive': ['bullish', 'growth', 'profit', 'gain', 'rise', 'increase', 'strong', 'outperform'],
    'negative': ['bearish', 'loss', 'decline', 'fall', 'decrease', 'weak', 'underperform', 'risk'],
    'neutral': ['stable', 'maintain', 'hold', 'unchanged', 'neutral']
}

def enhance_sentiment_analysis(text):
    """
    Enhance sentiment analysis with financial context
    """
    # Get base sentiment score
    sentiment = sia.polarity_scores(text)
    
    # Count financial keywords
    text_lower = text.lower()
    keyword_scores = {
        'positive': sum(1 for word in FINANCIAL_KEYWORDS['positive'] if word in text_lower),
        'negative': sum(1 for word in FINANCIAL_KEYWORDS['negative'] if word in text_lower),
        'neutral': sum(1 for word in FINANCIAL_KEYWORDS['neutral'] if word in text_lower)
    }
    
    # Adjust sentiment based on financial keywords
    sentiment['compound'] += (keyword_scores['positive'] * 0.1) - (keyword_scores['negative'] * 0.1)
    
    return sentiment

def get_market_impact(sentiment_score):
    """
    Determine market impact based on sentiment score
    """
    if sentiment_score >= 0.5:
        return "Strongly Positive"
    elif sentiment_score >= 0.2:
        return "Positive"
    elif sentiment_score >= -0.2:
        return "Neutral"
    elif sentiment_score >= -0.5:
        return "Negative"
    else:
        return "Strongly Negative"

def get_confidence_score(sentiment):
    """
    Calculate confidence score based on sentiment analysis
    """
    # Confidence is based on the absolute value of the compound score
    return min(100, int(abs(sentiment['compound']) * 100))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Perform sentiment analysis
        sentiment = enhance_sentiment_analysis(text)
        
        # Calculate additional metrics
        market_impact = get_market_impact(sentiment['compound'])
        confidence = get_confidence_score(sentiment)
        
        # Generate key insights
        key_insights = []
        if sentiment['compound'] > 0.2:
            key_insights.append("The text indicates positive market sentiment")
        elif sentiment['compound'] < -0.2:
            key_insights.append("The text indicates negative market sentiment")
        else:
            key_insights.append("The text indicates neutral market sentiment")
        
        # Add more specific insights based on sentiment components
        if sentiment['pos'] > 0.5:
            key_insights.append("Strong positive language detected")
        if sentiment['neg'] > 0.5:
            key_insights.append("Strong negative language detected")
        
        response = {
            'sentiment_score': round(sentiment['compound'], 2),
            'confidence': confidence,
            'market_impact': market_impact,
            'key_insights': key_insights,
            'sentiment_components': {
                'positive': round(sentiment['pos'], 2),
                'negative': round(sentiment['neg'], 2),
                'neutral': round(sentiment['neu'], 2)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 