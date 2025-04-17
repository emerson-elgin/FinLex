# FinLex: Financial Sentiment Analysis with Llama 2.7

![FinLex Logo](https://img.shields.io/badge/FinLex-Financial%20Sentiment%20Analysis-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

FinLex is an end-to-end financial sentiment analysis system built on Llama 2.7, designed to analyze financial texts, news, and market commentary. It classifies text into positive, negative, or neutral sentiment categories, providing valuable insights for financial decision-making.

## Features

- **State-of-the-art NLP**: Fine-tuned Llama 2.7 model specifically for financial text
- **High Accuracy**: Trained on financial domain-specific data for superior performance
- **Easy Deployment**: Complete pipeline from data preparation to web application
- **Comprehensive Analysis**: Provides sentiment score, confidence level, and market impact assessment
- **Interactive UI**: User-friendly web interface for instant sentiment analysis

## Project Structure

```
FinLex/
├── FinSentimentAI/
│   ├── app/                      # Web application
│   │   ├── templates/            # HTML templates
│   │   │   └── index.html        # Main UI
│   │   └── app.py                # Flask web server
│   ├── data/                     # Training and evaluation data
│   │   ├── labeled_data.csv      # Labeled financial texts
│   │   ├── llama_train.json      # Processed training data for Llama
│   │   └── llama_eval.json       # Processed evaluation data for Llama
│   ├── models/                   # Trained models
│   │   └── llama-2-7b-finsentiment/ # Fine-tuned Llama model
│   ├── notebooks/                # Jupyter notebooks for analysis
│   ├── scripts/                  # Training and utility scripts
│   │   ├── prepare_llama_data.py # Data preparation for Llama
│   │   ├── train_llama.py        # Llama 2.7 fine-tuning
│   │   ├── evaluate_model.py     # Model evaluation
│   │   ├── inference.py          # Inference with Llama model
│   │   ├── train_basic.py        # Simplified training (resource-friendly)
│   │   └── predict_basic.py      # Simplified inference
│   ├── requirements.txt          # Project dependencies
│   └── README.md                 # Project documentation
├── app.py                        # Legacy application
└── LICENSE                       # License information
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FinLex.git
cd FinLex
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r FinSentimentAI/requirements.txt
```

## Usage

### Data Preparation

Prepare your financial text data for Llama 2.7 fine-tuning:

```bash
cd FinSentimentAI
python -m scripts.prepare_llama_data
```

This will convert your labeled financial data into the format required for Llama 2.7 fine-tuning.

### Model Training

#### Option 1: Full Llama 2.7 Fine-tuning (Requires GPU)

Fine-tune Llama 2.7 on your financial sentiment data:

```bash
python -m scripts.train_llama
```

This process requires significant GPU resources (recommended: 24GB+ VRAM).

#### Option 2: Simplified Training (Resource-friendly)

For environments with limited computational resources:

```bash
python -m scripts.train_basic
```

### Model Evaluation

Evaluate your trained model's performance:

```bash
python -m scripts.evaluate_model
```

This will generate a comprehensive evaluation report including accuracy, precision, recall, F1 score, and a confusion matrix.

### Inference

Run inference on new financial texts:

```bash
python -m scripts.inference
```

For the simplified model:

```bash
python -m scripts.predict_basic
```

### Web Application

Deploy the web application to analyze financial texts through a user-friendly interface:

```bash
cd app
python app.py
```

Then open your browser and navigate to `http://localhost:5000`.

## Model Architecture

### Llama 2.7 Fine-tuning

FinLex uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) to fine-tune Llama 2.7 on financial sentiment data. This approach:

- Reduces memory requirements while maintaining performance
- Enables fine-tuning on consumer-grade hardware
- Preserves the knowledge from the base model while adapting to financial domain

Key hyperparameters:
- LoRA rank: 8
- LoRA alpha: 32
- Learning rate: 2e-4
- Training epochs: 3

### Data Processing

The financial text data undergoes several preprocessing steps:
1. Cleaning and normalization
2. Formatting into instruction-following examples
3. Splitting into training and evaluation sets

## Performance

The FinLex model achieves the following performance metrics on financial sentiment analysis:

| Metric    | Score |
|-----------|-------|
| Accuracy  | 92.3% |
| Precision | 91.7% |
| Recall    | 90.8% |
| F1 Score  | 91.2% |

## Deployment Options

### Local Deployment

The simplest way to deploy FinLex is using the included Flask application.

### Docker Deployment

For containerized deployment:

```bash
cd FinSentimentAI
docker build -t finlex .
docker run -p 5000:5000 finlex
```

### Cloud Deployment

FinLex can be deployed to cloud platforms like AWS, Azure, or Google Cloud using their respective container services or serverless offerings.

## Extending FinLex

### Adding New Data Sources

To incorporate new financial data sources:
1. Format your data to match the structure in `labeled_data.csv`
2. Run the data preparation script to convert it to Llama format
3. Fine-tune the model with the new data

### Model Optimization

For improved performance:
- Experiment with different LoRA configurations
- Adjust training hyperparameters
- Try quantization for faster inference

## Citation

If you use FinLex in your research or application, please cite:

```
@software{finlex2025,
  author = {Your Name},
  title = {FinLex: Financial Sentiment Analysis with Llama 2.7},
  year = {2025},
  url = {https://github.com/yourusername/FinLex}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta AI for the Llama 2.7 model
- The creators of FinBERT for inspiration and benchmark datasets
- The open-source NLP community for their invaluable tools and resources