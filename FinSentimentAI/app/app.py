import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Add scripts directory to path so we can import the predictor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

# Import our sentiment predictor
from predict_basic import FinancialSentimentPredictor

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Initialize sentiment predictor
predictor = FinancialSentimentPredictor()

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
        
        # Analyze sentiment
        result = predictor.predict(text, include_probabilities=True)
        
        # Map sentiment to market impact
        sentiment_score = result.get('confidence', 0.5)
        sentiment = result.get('sentiment', 'neutral')
        if sentiment == 'positive':
            impact_factor = 1
        elif sentiment == 'negative':
            impact_factor = -1
        else:
            impact_factor = 0
            
        score = impact_factor * sentiment_score
        
        # Determine market impact
        if score >= 0.5:
            market_impact = "Strongly Positive"
        elif score >= 0.2:
            market_impact = "Positive"
        elif score >= -0.2:
            market_impact = "Neutral"
        elif score >= -0.5:
            market_impact = "Negative"
        else:
            market_impact = "Strongly Negative"
        
        # Generate key insights
        key_insights = []
        if sentiment == 'positive':
            key_insights.append("The text indicates positive market sentiment")
            if result.get('confidence', 0) > 0.7:
                key_insights.append("High confidence in positive sentiment prediction")
        elif sentiment == 'negative':
            key_insights.append("The text indicates negative market sentiment")
            if result.get('confidence', 0) > 0.7:
                key_insights.append("High confidence in negative sentiment prediction")
        else:
            key_insights.append("The text indicates neutral market sentiment")
            
        # Prepare probabilities for sentiment components
        probs = result.get('probabilities', {})
        
        response = {
            'sentiment_score': round(score, 2),
            'confidence': int(result.get('confidence', 0.5) * 100),
            'market_impact': market_impact,
            'key_insights': key_insights,
            'sentiment_components': {
                'positive': round(probs.get('positive', 0), 2),
                'negative': round(probs.get('negative', 0), 2),
                'neutral': round(probs.get('neutral', 0), 2)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Make sure the templates directory exists
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(template_dir, exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
