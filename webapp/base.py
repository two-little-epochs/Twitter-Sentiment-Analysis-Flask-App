# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:53:52 2019

@author: YQ
"""

import json
import tweepy
import re
import pickle
import time
import pandas as pd

from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = pickle.load(open("tokenizer.p", "rb"))
model = load_model("twitter.h5")

def preprocess(text):
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        tokens.append(token)
    return " ".join(tokens)

def decode_sentiment(score, include_neutral=True):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    SENTIMENT_THRESHOLDS = (0.4, 0.7)

    if include_neutral:
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE
        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict(tweet_text, include_neutral=True):
    SEQUENCE_LENGTH = 300
    start_at = time.time()

    # Tokenize tweet_text
    processed_tweet_text = preprocess(tweet_text)
    word_vec = pad_sequences(tokenizer.texts_to_sequences([processed_tweet_text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([word_vec], batch_size=512)
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label,
            "score": float(score),
            "elapsed_time": time.time()-start_at}

# return timeline in list
def get_user_timeline(username):
    """
    parameters:
    username: string

    returns:
    api.user_timeline(username): list
    """

    consumer_key = "6Q0ILAuN9SYNo7QyE4FUYICPg"
    consumer_secret = "ehSKg8YwjdDlrI3qdjnw4qcc13pxZ3n98N7LJh4Urc4iHv1r48"
    access_key = "744994297-HpJhG5WiHFmexycz1iaKeeRHrbtOyyLWUyfGVjBw"
    access_secret = "mBWkFrZuJC4sV2jbWg7b5QvWm1HTUowwDuC25wSqnjQLz"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    return api.user_timeline(username)

def get_analyzed_tweets(timeline, start, end):
    """
    parameters:
    timeline: list
    start, end: string

    returns:
    analyzed_tweets: pandas.DataFrame
    """
    # process start and end date
    start = list(map(int, start.split('-')))
    start = datetime(start[0], start[1], start[2], 0, 0, 0)
    end = list(map(int, end.split('-')))
    end = datetime(end[0], end[1], end[2], 23, 59, 59)

    # load timeline obtained from twitter API (comes in some object format)
    json_timeline = timeline

    # populate analyzed tweets between stated dates into a Pandas DataFrame
    analyzed_tweets = pd.DataFrame(columns=["Datetime", "Tweet", "Sentiment", "Score"])
    for tweet in json_timeline:
        if tweet.created_at >= start and tweet.created_at <= end:
            prediction = predict(tweet.text)
            analyzed_tweets["Datetime"].append(tweet.created_at)
            analyzed_tweets["Tweet"].append(preprocess(tweet.text))
            analyzed_tweets["Sentiment"].append(prediction["label"])
            analyzed_tweets["Score"].append(prediction["score"])
        elif tweet.created_at < start:
            break

    return analyzed_tweets
