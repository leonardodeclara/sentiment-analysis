# Advanced Sentiment Analysis Microservice

This microservice provides sentiment analysis, emotion detection, and aspect-based sentiment analysis using Hugging Face models and FastAPI.

## Features
- **Sentiment Analysis**: Classifies text as positive or negative.
- **Emotion Detection**: Detects top 3 emotions in the text.
- **Aspect-Based Sentiment Analysis**: Analyzes sentiment for specific aspects in the text.
- **Rate Limiting**: Limits requests to 5 per minute.
- **Logging**: Logs all requests for monitoring.
- **Docker Support**: Containerized for easy deployment using Docker.

## Endpoints

### 1. Analyze Text
- **Endpoint**: `POST /analyze`
- **Description**: Analyzes the input text for sentiment, emotions, and aspects.
- **Request Body**:
  ```json
  {
    "text": "I love this product, but the delivery was late."
  }
  ```
- **Response**:
```json
{
  "text": "i love this product but the delivery was late",
  "sentiment": {
    "sentiment": "positive",
    "confidence": 0.987
  },
  "emotions": [
    {"emotion": "joy", "confidence": 0.95},
    {"emotion": "annoyance", "confidence": 0.75},
    {"emotion": "neutral", "confidence": 0.60}
  ],
  "aspects": [
    {"aspect": "product", "sentiment": 0.9},
    {"aspect": "delivery", "sentiment": 0.2}
  ]
}
```

### 2. Root Endpoint
- **Endpoint**: `GET /`
- **Description**: Returns a welcome message.
- **Response**:
  ```json
  {
    "message": "Advanced Sentiment Analysis Microservice"
  }

## Requirements
- Python 3.9
- Docker