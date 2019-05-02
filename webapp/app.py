# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:29:37 2019

@author: Yee Jet Tan
"""

from flask import Flask, render_template, request
<<<<<<< HEAD
import base

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    username = request.form['username']
    start = request.form['from']
    end = request.form['to']
    table = base.get_analyzed_tweets(base.get_user_timeline(username), start, end)
    return render_template('index.html', table=table.to_html())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
=======
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime

import re
import tweepy
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf


app = Flask(__name__)

tokenizer = pickle.load(open("tokenizer.p", "rb"))
model = load_model("twitter.h5")

# https://github.com/keras-team/keras/issues/10431
#model._make_predict_function()
graph = tf.get_default_graph()


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

    # populate analyzed tweets between stated dates into a Pandas DataFrame
    analyzed_tweets = pd.DataFrame(columns=["Datetime", "Tweet", "Sentiment", "Score"])
    for tweet in timeline:
        if tweet.created_at >= start and tweet.created_at <= end:
            prediction = predict(tweet.text)
            analyzed_tweets["Datetime"].append(tweet.created_at)
            analyzed_tweets["Tweet"].append(preprocess(tweet.text))
            analyzed_tweets["Sentiment"].append(prediction["label"])
            analyzed_tweets["Score"].append(prediction["score"])
        elif tweet.created_at < start:
            break

    return analyzed_tweets


def predict(timeline, start, end):
    # process start and end date
    start = list(map(int, start.split('-')))
    start = datetime(start[0], start[1], start[2], 0, 0, 0)
    end = list(map(int, end.split('-')))
    end = datetime(end[0], end[1], end[2], 23, 59, 59)
    SEQUENCE_LENGTH = 300

    analyzed_tweets = pd.DataFrame(columns=["Datetime", "Tweet", "Sentiment", "Score"])
    date = []
    tweets = []
    for tweet in timeline:
        if tweet.created_at >= start and tweet.created_at <= end:
            date.append(tweet.created_at)
            tweets.append(preprocess(tweet.text))
        elif tweet.created_at < start:
            break

    with graph.as_default():
        word_vec = pad_sequences(tokenizer.texts_to_sequences(tweets), maxlen=SEQUENCE_LENGTH)
        predictions = model.predict(word_vec, batch_size=8)

    analyzed_tweets["Score"] = np.squeeze(predictions, -1)
    analyzed_tweets["Sentiment"] = [decode_sentiment(x) for x in np.squeeze(predictions, -1)]
    analyzed_tweets["Datetime"] = date
    analyzed_tweets["Tweet"] = tweets

    return analyzed_tweets


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        start = request.form['from']
        end = request.form['to']

        timeline = get_user_timeline(username)
        table = predict(timeline, start, end)

        return render_template('index.html', tables=[table.to_html()], titles=table.columns.values)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
>>>>>>> fix
