# import dependencies
from flask import Flask, jsonify, render_template, redirect, request
import requests
import json
import time
import random
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from itertools import groupby
# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestClassifier
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Read in data
data = pd.read_csv('static/assets/data/compressed_df.csv').drop('Unnamed: 0', axis=1)
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
column_trans = ColumnTransformer(
    [('system_category', OneHotEncoder(dtype='int'), ['systems']),
     ('genre_category', OneHotEncoder(dtype='int'), ['genres']),
     ('playModes_category', OneHotEncoder(dtype='int'), ['playModes']),
     ('themes_category', OneHotEncoder(dtype='int'), ['themes']),
     ('series_category', OneHotEncoder(dtype='int'), ['series']),
     ('playerPerspectives', OneHotEncoder(dtype='int'), ['playerPerspectives']),
     ('TfIdf',TfidfVectorizer(stop_words='english'), 'gameDescription')],
    remainder='drop')

#prediction function
def ValuePredictor(to_predict_list):
    clf = pickle.load(open("static/assets/model.pkl","rb"))
    result = clf.predict(to_predict_list)
    return result[0]

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
def form_render():
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
def form_submit():
    start_time = time.time()
        
    column_trans.fit(X_df)
    X = column_trans.transform(X_df).toarray()

    # Retrieve data from HTML form
    data_row = []
    data_row.append(request.form['consoles'])
    data_row.append(request.form['genres'])
    data_row.append(request.form['playModes'])
    data_row.append(request.form['themes'])
    data_row.append(request.form['series'])
    data_row.append(request.form['perspectives'])
    data_row.append(request.form['text'])
    X_user = pd.DataFrame(np.array(data_row).reshape(1, -1), columns=['systems', 'genres', 'playModes', 'themes', 'series', 'playerPerspectives', 'gameDescription'])
    # transform user input data
    X_user_feed = column_trans.transform(X_user).toarray()
    # Predict
    prediction = ValuePredictor(X_user_feed)
    # BRUTE FORCE CLOSEST POINTS
    # compute distances
    d = ((X - X_user_feed)**2).sum(axis=1)
    ndx = d.argsort() # indirect sort 
    result = data.iloc[ndx[:10]]
    # print 10 nearest points to the chosen one
    print(result)
    similarGames = {}
    for i in range(len(result)):
        game_number = f"game_{i+1}"
        game_vars = []
        game_vars.append(result.iloc[i]['title'])
        game_vars.append(result.iloc[i]['moreInfo'])
        game_vars.append(result.iloc[i]['systems'])
        game_vars.append(result.iloc[i]['genres'])
        game_vars.append(result.iloc[i]['playModes'])
        game_vars.append(result.iloc[i]['themes'])
        game_vars.append(result.iloc[i]['series'])
        game_vars.append(result.iloc[i]['playerPerspectives'])
        game_vars.append(result.iloc[i]['imageLink'])
        game_vars.append(int(result.iloc[i]['memberRating']))
        game_vars.append(result.iloc[i]['gameDescription'])
        similarGames[game_number] = game_vars
    # Create Frequency counts of all ratings
    y_freq = [len(list(group)) for key, group in groupby(sorted(y.astype(int)))]
    summaryData = {}
    for i in range(len(y_freq)):
        summaryData[f"{i + 1}"] = y_freq[i]
    output = {
        "yourSystem": data_row[0],
        "yourGenres": data_row[1],
        "yourPlayModes": data_row[2],
        "yourThemes": data_row[3],
        "yourSeries": data_row[4],
        "yourPerspectives": data_row[5],
        "yourDesc": data_row[6],
        "prediction": int(prediction),
        "closestPoint": similarGames,
        "allRatings": summaryData
    }
    print("--- %s seconds ---" % (time.time() - start_time))
    return render_template("response.html", data=output)

if __name__ == "__main__":
    app.run(debug=True)
