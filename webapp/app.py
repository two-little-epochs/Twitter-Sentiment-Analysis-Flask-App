# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:29:37 2019

@author: Yee Jet Tan
"""

from flask import Flask, render_template, request
from pandas import DataFrame as df
import base

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        start = request.form['from']
        end = request.form['to']
        table = base.get_analyzed_tweets(base.get_user_timeline(username), start, end)
        return render_template('index.html', table=table.to_html())
    else:
        return render_template('index.html')

# @app.route('/process', methods=['POST'])
# def process():
#     username = request.form['username']
#     start = request.form['from']
#     end = request.form['to']
#     table = base.get_analyzed_tweets(base.get_user_timeline(username), start, end)
#     return render_template('index.html', table=table.to_html())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
