# author: Karanpal	Singh, Sreejith	Munthikodu, Sirine	Chahma
# date: 2020-01-22

'''This script downloads the data from a given url and saves in the data
folder in the project directory. This script takes a url to the data and a 
file location as the arguments.

Usage: get_data.py [--url=<url> --file_location=<file_location>]
 
'''

import requests
from docopt import docopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

import time

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import RFECV

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score




def main():
    # Load training data
    medical_data = pd.read_csv("../../data/processed/medical_cost_data_training.csv")
    # Split response and target variables
    X = medical_data.drop("charges", axis=1)
    y = medical_data.charges
    # Identify categoric and numeric features
    categoric_features, numeric_features = get_cat_num_features(X)

    # Create a preprocessing pipeline
    numeric_transformer = Pipeline(steps=[("poly" , PolynomialFeatures(degree=1)),
                                        ("scaler", StandardScaler())])
    categoric_transformer = Pipeline(steps=[("ohe", OneHotEncoder())])

    preprocessor = ColumnTransformer(transformers = [
                                    ("num", numeric_transformer, numeric_features),
                                    ("cat", categoric_transformer, categoric_features)])

    # Try various regression models
    models = {
        "linear_regression" : LinearRegression(),
        "decision_tree_regressor" : DecisionTreeRegressor(),
        "knn_regression" : KNeighborsRegressor(),
        "rf_regression" : RandomForestRegressor(),
        "SVR" : SVR()   
    }

    # Build a RandomForestRegressor model
    reg = RandomForestRegressor(max_depth=5, min_samples_split=4)
    model = Pipeline(steps = [("preprocessor", preprocessor),
                            ("regressor", reg)])
    
    # Fit on the training data
    model.fit(X, y)

    # Get the training error
    y_predicted_training = model.predict(X)
    training_error = mean_absolute_error(y, y_predicted_training)

    # Load the test data
    medical_data_test = pd.read_csv("../../data/processed/medical_cost_data_test.csv")

    # Split response and target variables
    X_test = medical_data_test.drop("charges", axis=1)
    y_test = medical_data_test.charges

    # Get the test error
    y_predicted_test = model.predict(X_test)
    test_error = mean_absolute_error(y_test, y_predicted_test)

    # Print result
    print(f"Training error: {training_error}, \nTest error: {test_error}")
    


def get_cat_num_features(x):
    data_types = x.dtypes
    categoric_features = []
    numeric_features = []
    for d_type, feature in zip(data_types, data_types.index):
        if d_type == "object":
            categoric_features.append(feature)
        else:
            numeric_features.append(feature)
    return categoric_features, numeric_features

if __name__ == "__main__":
    opt = docopt(__doc__)
    main()