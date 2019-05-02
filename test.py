# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:24:04 2019

@author: YQ
"""

import json
import re
import time
import pickle
import pandas as pd

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = pickle.load(open("tokenizer.p", "rb"))
model = keras.models.load_model("twitter.h5")

SEQUENCE_LENGTH = 300
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        tokens.append(token)
    return " ".join(tokens)

def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    text = preprocess(text)
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}

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

file = json.load(open("starbizmy_short.json", "rb"))

df = pd.DataFrame(columns=["tweet", "sentiment", "score"])
df["tweet"] = [preprocess(x["text"]) for x in file]

x = pad_sequences(tokenizer.texts_to_sequences(df["tweet"]), maxlen=SEQUENCE_LENGTH)
score = model.predict(x, batch_size=512)
sentiment = [decode_sentiment(i) for i in score]

df["sentiment"] = sentiment
df["score"] = score

df.to_csv("starbiz_sentiment.csv", encoding='utf-8')
