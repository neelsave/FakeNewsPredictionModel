from flask import Flask,render_template,abort, jsonify, request
from flask_cors import CORS
import os
from sklearn.externals import joblib
from feature import *
import json
import numpy as np
import pickle
import flask
import newspaper
from newspaper import Article
import urllib

pipeline2 = joblib.load("C:\\Users\\admin\\Desktop\\Merging_Link_and_box\\Model1.sav")

app = Flask(__name__)
CORS(app)


with open('Link_Model.pickle', 'rb') as handle:
	model = pickle.load(handle)


@app.route("/")
def home():
    return render_template('index.html')

@app.route('/index2',methods=['POST'])
def get_delay():
    result=request.form
    query_title = result['title']
    query_author = result['author']
    query_text = result['maintext']
    news_link = result['link']
    if news_link != "":
        url = urllib.parse.unquote(news_link)
        article = Article(str(url))
        article.download()
        article.parse()
        article.nlp()
        news = article.summary
        pred = model.predict([news])
        if(pred == ['REAL']):
            return render_template('index2.html',prediction_text='The news is Mostly Real',query_title=query_title,query_author=query_author,query_text=news)
        else:
            return render_template('index3.html',prediction_text='The news is Mostly Fake ',query_title=query_title,query_author=query_author,query_text=news)
    total= query_title       
    query = [total]
    pred = pipeline2.predict(query)
    if(pred == [0]):
        return render_template('index2.html',prediction_text='The news is Mostly Real',query_title=query_title,query_author=query_author,query_text=query_text)
    else:
        return render_template('index3.html',prediction_text='The news is Mostly Fake ',query_title=query_title,query_author=query_author,query_text=query_text)


    

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/post")
def post():
    return render_template('post.html')

app.run(debug=True)