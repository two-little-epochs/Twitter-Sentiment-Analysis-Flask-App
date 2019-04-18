# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:53:52 2019

@author: YQ
"""

# https://stackoverflow.com/questions/49731259/tweepy-get-tweets-among-two-dates/49731413

import tweepy
import re
import pickle
import time
import pandas as pd

from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEQUENCE_LENGTH = 300
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

tokenizer = pickle.load(open("tokenizer.p", "rb"))
model = load_model("twitter.h5")

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
    
    
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

"""
User input arguments
"""
timeline = api.user_timeline("fchollet")

stop = datetime(2019, 4, 18, 23, 59, 59)
start = datetime(2019, 4, 10, 0, 0, 0)
########################################


tweets = []
for tweet in timeline:
    if tweet.created_at < stop and tweet.created_at >= start:
        prediction = predict(tweet.text)
        tweets.append((tweet.created_at, preprocess(tweet.text), prediction["label"], prediction["score"]))
    elif tweet.created_at < start:
        break

tweets = pd.DataFrame(tweets, columns=["Datetime", "Tweet", "Sentiment", "Score"])

