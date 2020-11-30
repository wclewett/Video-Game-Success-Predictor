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




app = Flask(__name__)

# home page
@app.route("/")
def homepage():
    
    return render_template("layout.html", data=dummy)

# results page
@app.route("/", methods=['POST'])
def game_description():
    start_time = time.time()
    # Retrieve data from HTML form
    text = []
    text.append(request.form['text'])
    console_family = []
    console_family.append(request.form['consoles'])
    genres = []
    genres.append(request.form['genres'])
    themes = []
    themes.append(request.form['themes'])
    perspectives = []
    perspectives.append(request.form['perspective'])    
    print(text)
    print(console_family)
    print(genres)
    print(themes)
    print(perspectives)
    vectors_test = vectorizer.transform(text)
    # Predict
    predicted = rf_model.predict(vectors_test)
    output = {
        "yourDesc": text[0],
        "yourConsole": console_family[0],
        "yourGenres": genres[0],
        "yourThemes": themes[0],
        "yourPerspective": perspectives[0],
        "output": int(predicted[0])
    }
    print(output)
    print("--- %s seconds ---" % (time.time() - start_time))
    return render_template("application.html", data=output)

if __name__ == "__main__":
    app.run(debug=True)