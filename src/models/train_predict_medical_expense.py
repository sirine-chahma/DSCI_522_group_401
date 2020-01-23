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

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score




def main():
    # Load training data
    print("Loading preprocessed training data...\n")
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
    print("Trying various regression models...\n")
    models = {
    "linear_regression" : LinearRegression(),
    "decision_tree_regressor" : DecisionTreeRegressor(),
    "knn_regression" : KNeighborsRegressor(),
    "rf_regression" : RandomForestRegressor(n_estimators=100),
    "SVR" : SVR(gamma="scale")   
    }

    results_df = try_models(models, X, y, preprocessor)
    print(results_df)

    # Tune DecisionTreeRegressor
    print("\nPerforming hyper-parameter tuning on DecisionTreeRegressor. This may take several minutes...\n")
    reg = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', DecisionTreeRegressor())])

    param_grid = {
        'preprocessor__num__poly__degree': list(range(1, 5)),
        'regressor__max_depth': list(range(1, 20)),
        "regressor__min_samples_split" : list(range(2, 10))
    }

    grid = GridSearchCV(reg, param_grid, cv=5, 
                        return_train_score=True, 
                        scoring="neg_mean_absolute_error",
                        iid=False)
    grid.fit(X, y)

    print(f"Best parameters are : {grid.best_params_}\n")

    # Load the test data
    print("Loading preprocessed test data...\n")
    medical_data_test = pd.read_csv("../../data/processed/medical_cost_data_test.csv")

    # Split response and target variables
    X_test = medical_data_test.drop("charges", axis=1)
    y_test = medical_data_test.charges

    # Get the train and test error
    print("Testing the model on training and test data...\n")
    train_errors = get_error(grid, X, y)
    test_errors = get_error(grid, X_test, y_test)

    final_results = {"training data" : train_errors,
                    "test data" : test_errors}
    results_df = pd.DataFrame(final_results)
    results_df.index = ["Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "R2_score", "Explained_variance_score"]
    print("Results: \n")
    print(results_df)

    # Plot prediction results
    print("Generating plots...\n")
    y_train_predicted = grid.predict(X)
    y_test_predicted = grid.predict(X_test)
    prediction_df = pd.DataFrame({"y_actual" : y_test, "y_predicted" : y_test_predicted, "residuals" : y_test - y_test_predicted})

    # Predicted vs actual
    predicted_vs_actual_plot = alt.Chart(prediction_df).mark_circle(size=60, color="green", opacity=0.4).encode(
                                    alt.X('y_predicted', title = "Predicted Medical Expense ($)"),
                                    alt.Y('y_actual', title = "Actual Medical Expense ($)") 
                                    ).properties(
                                    title = "Predicted Vs Actual",
                                    width = 700,
                                    height = 300
                                    ).configure_axis(
                                    labelFontSize = 20,
                                    titleFontSize = 20
                                    ).configure_title(
                                    fontSize = 20
                                    )
    # Residual plot
    residual_plot = alt.Chart(prediction_df).mark_circle(size=60, color="red", opacity=0.4).encode(
                        alt.X('y_predicted', title = "Predicted Medical Expense ($)"),
                        alt.Y('residuals', title = "Residuals ($)") 
                        ).properties(
                        title = "Residual Plot",
                        width = 700,
                        height = 300
                        ).configure_axis(
                        labelFontSize = 20,
                        titleFontSize = 20
                        ).configure_title(
                        fontSize = 20
                        )
    # Save the plots
    print("Saving generated plots...\n")
    predicted_vs_actual_plot.save("predicted_vs_actual_plot.png")
    residual_plot.save("residual_plot.png")


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


def try_models(models, X_train, y_train, preprocessor):
    results_dict = {}
    for model_name, model in models.items():
        # print(f"trainning {model}")
        t = time.time()
        reg = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', model)])
        validation_error = np.mean(cross_val_score(reg, X_train, 
                                                   y_train, 
                                                   scoring = "neg_mean_absolute_error", 
                                                   cv=5) * -1)
        reg.fit(X_train, y_train)
        tr_err = mean_absolute_error(y_train, reg.predict(X_train))
        elapsed_time = time.time() - t
        results_dict[model_name] = [np.round(tr_err,3), np.round(validation_error,3), np.round(elapsed_time,4)]
        indices = ["Mean absolute error, training", "Mean absolute error, test", "training_time(s)"]
    results_df = pd.DataFrame(results_dict)
    results_df.index = indices

    return results_df


def get_error(model, X, y):
    y_predicted = model.predict(X)
    mean_abs_error = mean_absolute_error(y, y_predicted)
    mean_sqrd_error = mean_squared_error(y, y_predicted)
    root_mean_sqrd_error = np.sqrt(mean_squared_error(y, y_predicted))
    r2 = r2_score(y, y_predicted)
    exp_var_score = explained_variance_score(y, y_predicted)

    return [mean_abs_error, mean_sqrd_error, root_mean_sqrd_error, r2, exp_var_score]


if __name__ == "__main__":
    opt = docopt(__doc__)
    main()