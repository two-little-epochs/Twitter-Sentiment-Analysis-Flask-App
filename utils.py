import time
import re
import numpy as np
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stemmer = SnowballStemmer("english")
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if stem:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    return " ".join(tokens)
    
    
def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE    
    
    
def predict(model, tokenizer, tweets, include_neutral=True, sequence_length=300):
    """
    model: Keras model
    tokenizer: Tokenizer object
    tweets: List of strings (tweets)
    """
    
    start_at = time.time()
    # Tokenize text
    tweets = [preprocess(t) for t in tweets]
    X = pad_sequences(tokenizer.texts_to_sequences(tweets), maxlen=sequence_length)
    # Predict
    score = model.predict(X, batch_size=50)
    # Decode sentiment
    labels = []
    for s in score:
        label = decode_sentiment(s, include_neutral=include_neutral)
        labels.append(label)

    return {"label": labels, "score": score,
       "elapsed_time": time.time()-start_at}      
