# -*- coding: utf-8 -*-

from utils import predict
from tensorflow.keras.models import load_model
from datetime import datetime
from flask import Flask, render_template, request
from tensorflow.python.keras.backend import set_session

import json
import tweepy
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd

app = Flask(__name__)
keys = json.load(open("keys.json"))
tokenizer = pickle.load(open("tokenizer.p", "rb"))

global graph  # https://towardsdatascience.com/deploying-keras-deep-learning-models-with-flask-5da4181436a2
graph = tf.get_default_graph()
sess = tf.Session()
set_session(sess)  # https://github.com/tensorflow/tensorflow/issues/28287
model = load_model("twitter.h5")

def get_user_timeline(username):
    """
    parameters:
    username: string
    returns:
    api.user_timeline(username): list
    """

    auth = tweepy.OAuthHandler(keys["consumer_key"], keys["consumer_secret"])
    auth.set_access_token(keys["access_key"], keys["access_secret"])
    api = tweepy.API(auth)

    #return api.user_timeline(username)
    return tweepy.Cursor(api.user_timeline, screen_name=username, tweet_mode="extended").items()

def make_prediction(timeline, start, end):
    # process start and end date
    start = list(map(int, start.split('-')))
    start = datetime(start[0], start[1], start[2], 0, 0, 0)
    end = list(map(int, end.split('-')))
    end = datetime(end[0], end[1], end[2], 23, 59, 59)

    analyzed_tweets = pd.DataFrame(columns=["Datetime", "Tweet", "Sentiment", "Score"])
    date = []
    tweets = []
    for tweet in timeline:
        if tweet.created_at >= start and tweet.created_at <= end:
            date.append(tweet.created_at)
            try:
                if tweet.retweeted_status:
                    tweets.append("RT " + tweet.retweeted_status.full_text)
            except:
                tweets.append(tweet.full_text)
        elif tweet.created_at < start:
            break

    with graph.as_default():
        set_session(sess)
        predictions = predict(model, tokenizer, tweets)

    print(predictions["score"].shape)
    analyzed_tweets["Score"] = np.squeeze(predictions["score"], -1)
    analyzed_tweets["Sentiment"] = predictions["label"]
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
        table = make_prediction(timeline, start, end)

        return render_template('index.html', tables=[table.to_html()], titles=table.columns.values)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
