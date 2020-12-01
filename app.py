# import dependencies
from flask import Flask, jsonify, render_template, redirect, request
import requests
import json
import time
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Read in data
data = pd.read_csv('assets/data/compressed_df.csv').drop('Unnamed: 0', axis=1)

# begin X,y transformation
ml_data_feed = data[['systems', 'genres', 'playModes', 'themes', 'series', 'playerPerspectives', 'gameDescription', 'memberRating']]
ml_data_feed = ml_data_feed.fillna('None Specified')
X_df = ml_data_feed[['systems', 'genres', 'playModes', 'themes', 'series', 'playerPerspectives', 'gameDescription']]

# Create X,y
y = ml_data_feed['memberRating']
for i in range(len(y)):
    if y[i] > 85:
        y[i] = 5
    elif y[i] > 75:
        y[i] = 4
    elif y[i] > 65:
        y[i] = 3    
    elif y[i] > 55:
        y[i] = 2
    else:
        y[i] = 1
y.values.reshape(-1, 1)

# Setup Pipeline
column_trans = ColumnTransformer(
    [('system_category', OneHotEncoder(dtype='int'), ['systems']),
     ('genre_category', OneHotEncoder(dtype='int'), ['genres']),
     ('playModes_category', OneHotEncoder(dtype='int'), ['playModes']),
     ('themes_category', OneHotEncoder(dtype='int'), ['themes']),
     ('series_category', OneHotEncoder(dtype='int'), ['series']),
     ('playerPerspectives', OneHotEncoder(dtype='int'), ['playerPerspectives']),
     ('TfIdf',TfidfVectorizer(stop_words='english'), 'gameDescription')],
    remainder='drop')

column_trans.fit(X_df)
X = column_trans.transform(X_df).toarray()
print(X)


# Split data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

#####################
### RANDOM FOREST ###
#####################

# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10)

# Train
rf.fit(X_train, y_train.astype(int))

# home page
@app.route("/")
def homepage():
    data_length = len(data)
    ran_nums = []
    for i in range(10):
          r = random.randint(0, data_length)
          if r not in ran_nums:
            ran_nums.append(r)
    image_link_1 = data['imageLink'][ran_nums[0]]
    title_1 = data['title'][ran_nums[0]]
    description_1 = data['gameDescription'][ran_nums[0]]
    image_link_2 = data['imageLink'][ran_nums[1]]
    title_2 = data['title'][ran_nums[1]]    
    description_2 = data['gameDescription'][ran_nums[1]]
    image_link_3 = data['imageLink'][ran_nums[2]]
    title_3 = data['title'][ran_nums[2]]
    description_3 = data['gameDescription'][ran_nums[2]]
    payload = {
        'imageLeft': image_link_1,
        'descLeft': description_1,
        'titleLeft': title_1,
        'imageMiddle': image_link_2,
        'descMiddle': description_2,
        'titleMiddle': title_2,
        'imageRight': image_link_3,
        'descRight': description_3,
        'titleRight': title_3  
    }
    return render_template("home.html", data=payload)

# Application Page
@app.route("/application")
def form_submit():
    unique_systems = list(ml_data_feed['systems'].unique())
    unique_genres = sorted(list(ml_data_feed['genres'].unique()))
    unique_playModes = sorted(list(ml_data_feed['playModes'].unique()))
    unique_themes = sorted(list(ml_data_feed['themes'].unique()))
    unique_series = sorted(list(ml_data_feed['series'].unique()))
    unique_perspective = sorted(list(ml_data_feed['playerPerspectives'].unique()))
    payload = {
        'systems': unique_systems,
        'genres': unique_genres,
        'playModes': unique_playModes,
        'themes': unique_themes,
        'series': unique_series,
        'playerPerspectives': unique_perspective
    }
    return render_template("application.html", data=payload)

# results page
@app.route("/application", methods=['POST'])
def game_description():
    start_time = time.time()
    # Retrieve data from HTML form
    console_family = []
    console_family.append(request.form['consoles'])
    genres = []
    genres.append(request.form['genres'])
    playModes = []
    playModes.append(request.form['playModes'])
    themes = []
    themes.append(request.form['themes'])
    series = []
    series.append(request.form['series'])
    perspectives = []
    perspectives.append(request.form['perspective'])
    text = []
    text.append(request.form['text'])
    print(text)
    print(console_family)
    print(genres)
    print(themes)
    print(perspectives)
    # transform user input data

    # Predict

    # BRUTE FORCE CLOSEST POINTS
    similarGames = {
        "game_1": game_1,
        "game_2": game_2,
        "game_3": game_3,
        "game_4": game_4,
        "game_5": game_5,
        "game_6": game_6,
        "game_7": game_7,
        "game_8": game_8,
        "game_9": game_9,
        "game_10": game_10
    }
    output = {
        "yourDesc": text[0],
        "yourConsole": console_family[0],
        "yourGenres": genres[0],
        "yourThemes": themes[0],
        "yourPerspective": perspectives[0],
        "output": int(predicted[0]),
        "similarGames": similarGames
    }
    print(output)
    print("--- %s seconds ---" % (time.time() - start_time))
    return render_template("response.html", data=output)

if __name__ == "__main__":
    app.run(debug=True)