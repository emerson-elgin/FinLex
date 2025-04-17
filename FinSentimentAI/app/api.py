import os
import sys
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add scripts directory to path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

# Import our sentiment analyzer
from inference import FinancialSentimentAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "../logs/api.log"), mode="a")
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), "../logs"), exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="API for financial sentiment analysis using fine-tuned Llama 2.7",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sentiment analyzer
analyzer = None

class SentimentRequest(BaseModel):
    text: str
    include_reasoning: bool = False

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float = None
    reasoning: str = None
    
@app.on_event("startup")
async def startup_event():
    global analyzer
    logger.info("Initializing sentiment analyzer...")
    
    try:
        analyzer = FinancialSentimentAnalyzer()
        logger.info("Sentiment analyzer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API...")

@app.get("/")
async def root():
    return {"message": "Financial Sentiment Analysis API is running"}

@app.post("/analyze", response_model=dict)
async def analyze_sentiment(request: SentimentRequest):
    if not analyzer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Analyzing sentiment for text: {request.text[:50]}...")
        result = analyzer.analyze_sentiment(request.text, include_reasoning=request.include_reasoning)
        
        # Add confidence measure (simplified)
        sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
        response = {
            "sentiment": result["sentiment"],
            "confidence": 0.8,  # Placeholder; in a real app this would be from the model
        }
        
        if request.include_reasoning and "reasoning" in result:
            response["reasoning"] = result["reasoning"]
        
        logger.info(f"Sentiment analysis complete: {response['sentiment']}")
        return response
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if not analyzer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
