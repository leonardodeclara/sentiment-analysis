from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# Load advanced sentiment analysis model
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)

# Load emotion detection model
EMOTION_MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-stable"
emotion_pipeline = pipeline("text-classification", model=EMOTION_MODEL_NAME)

# Load aspect-based sentiment analysis model
ABSA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
absa_pipeline = pipeline("text-classification", model=ABSA_MODEL_NAME)

def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by lowercasing, removing special characters, and trimming extra spaces.

    Args:
        text (str): Input text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of the input text using a pre-trained sentiment analysis model.

    Args:
        text (str): Input text to analyze.

    Returns:
        dict: Sentiment analysis result with 'sentiment' and 'confidence'.
    """
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    sentiment = "positive" if probs.argmax().item() == 1 else "negative"
    confidence = probs.max().item()
    return {"sentiment": sentiment, "confidence": confidence}

def detect_emotions(text: str) -> list:
    """
    Detect emotions in the input text using a pre-trained emotion detection model.

    Args:
        text (str): Input text to analyze.

    Returns:
        list: List of detected emotions with 'emotion' and 'confidence'.
    """
    emotions = emotion_pipeline(text, top_k=3)  # Get top 3 emotions
    return [{"emotion": e["label"], "confidence": e["score"]} for e in emotions]

def analyze_aspects(text: str) -> list:
    """
    Perform aspect-based sentiment analysis on the input text.

    Args:
        text (str): Input text to analyze.

    Returns:
        list: List of aspects with 'aspect' and 'sentiment'.
    """
    results = absa_pipeline(text)
    return [{"aspect": r["label"], "sentiment": r["score"]} for r in results]