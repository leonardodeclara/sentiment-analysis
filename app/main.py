from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from model import preprocess_text, analyze_sentiment, detect_emotions, analyze_aspects
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging

# Initialize FastAPI app
app = FastAPI()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request body schema
class TextInput(BaseModel):
    text: str

@app.post("/analyze")
@limiter.limit("5/minute")  # Apply rate limiting (5 requests per minute)
async def analyze(text_input: TextInput, request: Request):
    """
    Analyze the input text for sentiment, emotions, and aspects.

    Args:
        text_input (TextInput): Input text to analyze.
        request (Request): FastAPI request object for rate limiting.

    Returns:
        dict: Analysis results including sentiment, emotions, and aspects.
    """
    logger.info(f"Analyzing text: {text_input.text}")

    # Validate input text
    if not text_input.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    # Preprocess text
    text = preprocess_text(text_input.text)

    # Analyze sentiment
    sentiment_result = analyze_sentiment(text)

    # Detect emotions
    emotion_result = detect_emotions(text)

    # Analyze aspects
    aspect_result = analyze_aspects(text)

    return {
        "text": text,
        "sentiment": sentiment_result,
        "emotions": emotion_result,
        "aspects": aspect_result,
    }

@app.get("/")
async def root():
    """
    Root endpoint to provide a welcome message.

    Returns:
        dict: Welcome message.
    """
    return {"message": "Advanced Sentiment Analysis Microservice"}