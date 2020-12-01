from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

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
print(y.value_counts())
y.values.reshape(-1, 1)
    
print(X_df.shape, y.shape)


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
column_trans.get_feature_names()
X = column_trans.transform(X_df).toarray()
print(X)


# Split data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

#####################
### RANDOM FOREST ###
#####################

# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1500)

# Train
rf.fit(X_train, y_train.astype(int))
# Test
predicted = rf.predict(X_test)
pred_bins = np.bincount(predicted)
occurance = np.nonzero(pred_bins)[0]
print(" ")
print("Random Forest Guess Rates")
for i in range(len(occurance)):
    print(occurance[i], pred_bins[occurance][i]/len(y_test))

# Accuracy
print(f"The random forest is {np.round(np.mean(predicted == y_test.astype(int)) * 100, decimals=2)}% accurate.")

#####################
### DECISION TREE ###
#####################

# # Model
# clf = tree.DecisionTreeClassifier()

# # Train
# clf = clf.fit(X_train, y_train.astype(int))

# # Predict
# tree_pred = clf.predict(X_test)
# pred_bins = np.bincount(tree_pred)
# occurance = np.nonzero(pred_bins)[0]
# print(" ")
# print("Decision Tree Guess Rates")
# for i in range(len(occurance)):
#     print(occurance[i], pred_bins[occurance][i]/len(y_test))
# # Accuracy
# print(f"The decision tree is {np.round(np.mean(tree_pred == y_test.astype(int)) * 100, decimals=2)}% accurate.")

#################################
### SUPPORT VECTOR CLASSIFIER ###
#################################
### Removed for poor results ####
#################################


# from sklearn.svm import SVC 
# # Model
# svc_model = SVC(kernel='poly', gamma='auto')

# # Train
# svc_model = svc_model.fit(X_train, y_train)

# # Predict
# svc_pred = svc_model.predict(X_test)
# pred_bins = np.bincount(svc_pred)
# occurance = np.nonzero(pred_bins)[0]
# print(" ")
# print("SVC Guess Rates")
# for i in range(len(occurance)):
#     print(occurance[i], pred_bins[occurance][i]/len(y_test))
# # Accuracy
# print(f"The support vector classifier is {np.round(np.mean(tree_pred == y_test.astype(int)) * 100, decimals=2)}% accurate.")

###############################################
### KMeans for Similarity Recomendation Bot ###
###############################################
# from sklearn.neighbors import KNeighborsRegressor

# # Create the knn model.
# # Look at the five closest neighbors.
# knn = KNeighborsRegressor(n_neighbors=20)
# # Fit the model on the training data.
# knn.fit(X_train, y_train)
# # Make point predictions on the test set using the fit model.
# predictions = knn.predict(X_test)

# print(predictions)

#########################
### LINEAR REGRESSION ###
#########################
# from sklearn.linear_model import LinearRegression

# #Call Model
# linReg = LinearRegression()

# # Train
# linReg.fit(gameDescription_transformed, y_train.astype(int))

# # Predict
# predictions = linReg.predict(test_Tfidf_array)

# score = linReg.score(test_Tfidf_array, y_test)
# print(f"R2 Score: {score}")