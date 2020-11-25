# import dependencies
from flask import Flask, jsonify, render_template, redirect, request
import requests
import json
import time
import pymongo
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)

# home page
@app.route("/")
def index():
    dummy = {}
    return render_template("index.html", data=dummy)


# results page
@app.route("/", methods=['POST'])
def game_description():
    test_df = pd.read_csv('assets/data/cleaned_test_df.csv').drop('Unnamed: 0', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(test_df['text'].to_numpy(), test_df['rating'].to_numpy(),
                                                    test_size=0.75, random_state=42)
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    vectors_train = vectorizer.fit_transform(X_train)
    # Model
    rf = RandomForestClassifier(n_estimators=100)
    # Train
    y_train=y_train.astype(int)
    rf.fit(vectors_train, y_train)
    # Retrieve data from HTML form
    text = []
    text.append(request.form['text'])
    console_family = []
    console_family.append(request.form['consoles'])
    genres = []
    genres.append(request.form['genres'])
    themes = []
    themes.append(request.form['themes'])    
    print(text)
    print(console_family)
    print(genres)
    print(themes)
    vectors_test = vectorizer.transform(text)
    # Predict
    predicted = rf.predict(vectors_test)
    output = {
        "output": int(predicted[0])
    }
    print(output) 
    return render_template("index.html", data=output)

if __name__ == "__main__":
    app.run(debug=True)