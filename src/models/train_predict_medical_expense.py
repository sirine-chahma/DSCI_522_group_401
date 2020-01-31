# author: Karanpal	Singh, Sreejith	Munthikodu, Sirine	Chahma
# date: 2020-01-22

'''This reads training and test data from specified locations, perform 
predictive analysis by performing pre-processing, comparing various models,
hyper-parameter optimization of the selected model, and evaluating on the
training and test dataset. It saves the performance of various models,
performance of the final tuned model, predicted vs actual plot and predicted 
vs residuals plot. In the current version, this script is not robust enough
to select a model from the list of models and perform hyper-parameter tuning. 

Usage: train_predict_medical_expense.py --training_data_file_path=<training_data_file_path>
--test_data_file_path=<test_data_file_path> --results_file_location=<results_file_location>
 
'''

import requests
from docopt import docopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

import time
import os

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




def main(training_data_file_path, test_data_file_path, results_file_location):
    """The main function."""

    # Test all functions
    test_get_cat_num_features()
    test_try_models()
    test_get_error()

    # Load training data
    print("\nLoading preprocessed training data...\n")
    medical_data = pd.read_csv(training_data_file_path)

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

    # Save preprocessing details
    preprocessor_df = pd.DataFrame({"Numeric features" : ["PolynomialFeatures", "StandardScaler"],
                                    "Categorical features" : ["OneHotEncoder", ""]})
    preprocessor_df.to_csv(os.path.join(results_file_location, "tables/preprocessors.csv"))

    # Try various regression models
    print("Trying various regression models...\n")
    models = {
    "Linear Regression" : LinearRegression(),
    "Decision Tree Regressor" : DecisionTreeRegressor(),
    "KNN Regressor" : KNeighborsRegressor(),
    "Random Forest Regressor" : RandomForestRegressor(n_estimators=100),
    "SVR (Support Vector Regressor)" : SVR(gamma="scale")   
    }

    results_df = try_models(models, X, y, preprocessor)
    print("Results from fitting models with default parameters: \n")
    print(results_df)

    # Save performance of various models
    results_df.to_csv(os.path.join(results_file_location, "tables/regression_models_base_errors.csv"))


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
                        scoring="neg_mean_absolute_error")
    grid.fit(X, y)

    # Save best parameters
    best_params_df = pd.DataFrame.from_dict(grid.best_params_, "index")
    best_params_df.columns = ["Best parameter"]
    print(f"Best parameters are : {best_params_df}\n")
    best_params_df.to_csv(os.path.join(results_file_location, "tables/hyperparameters.csv"))

    # Load the test data
    print("Loading preprocessed test data...\n")
    medical_data_test = pd.read_csv(test_data_file_path)

    # Split response and target variables
    X_test = medical_data_test.drop("charges", axis=1)
    y_test = medical_data_test.charges

    # Get the train and test error
    print("Testing the model on training and test data...\n")
    train_errors = get_error(grid, X, y)
    test_errors = get_error(grid, X_test, y_test)

    final_results = {"training_data" : train_errors,
                    "test_data" : test_errors}
    results_df = pd.DataFrame(final_results)
    results_df.index = ["Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "R2_score", "Explained_variance_score"]
    print("Results: \n")
    print(results_df)

    # Save results
    results_df.to_csv(os.path.join(results_file_location, "tables/regression_errors.csv"))

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
    predicted_vs_actual_plot.save(os.path.join(results_file_location, "figures/predicted_vs_actual_plot.png"))
    residual_plot.save(os.path.join(results_file_location, "figures/residual_plot.png"))


def get_cat_num_features(x):
    """ Identifies categorical and numerical
    features from the columns of a dataframe 
    using data type. 
​
    Parameters
    ----------
    x : DateFrame
        Training data.
​
    Returns
    -------
    categoric_features, numeric_features : tuple
        Tuple containinig lists of categoric 
        feature names and numeric feature names
    """
    data_types = x.dtypes
    categoric_features = []
    numeric_features = []
    for d_type, feature in zip(data_types, data_types.index):
        if d_type == "object":
            categoric_features.append(feature)
        else:
            numeric_features.append(feature)

    return categoric_features, numeric_features

def test_get_cat_num_features():
    """
    Test for get_cat_num_features() function
    """
    # Generate some test data
    X = pd.DataFrame({"cat" : ["a", "b", "c"],
                    "num" : [1, 2, 3]})

    cat, num = get_cat_num_features(X)
    # Ensure output dimensions are as expected
    assert(len(cat) == 1)
    assert(len(num) == 1)

def try_models(models, X_train, y_train, preprocessor):
    """ Fits different regression models on the given
    dataset and evaluates the mean absolute error. 
​
    Parameters
    ----------
    models : dict
        Dictionary of various regression models to try.
    X_train : DataFrame
        Training data on which the regression models
        are to be fitted
    y_train: Series
        True labels of the training data
    preprocessor: ColumnTransformer
        Sklearn column transformer to preprocess the 
        categorical and numerical features
​
    Returns
    -------
    results_df : DataFrame
        Dataframe containing the mean absolute error
        of each of the regression models
    """
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
        indices = ["Mean absolute error, training", "Mean absolute error, validation", "training_time(s)"]
    results_df = pd.DataFrame(results_dict)
    results_df.index = indices

    return results_df

def test_try_models():
    """
    Test for try_models() function
    """
    models = {"lm" : LinearRegression()}
    # Generate some test data
    X = pd.DataFrame({"num1" : [1, 2, 3, 4, 5],
                    "num2" : [10, 26, 3, 18, 11],
                    "cat" : ["a", "b", "a", "b", "a"]})
    y = np.array([10, 2, 13, 10, 11])

    numeric_transformer = Pipeline(steps=[("poly" , PolynomialFeatures(degree=1)),
                                        ("scaler", StandardScaler())])
    categoric_transformer = Pipeline(steps=[("ohe", OneHotEncoder())])

    preprocessor = ColumnTransformer(transformers = [
                                    ("num", numeric_transformer, ["num1", "num2"]),
                                    ("cat", categoric_transformer, ["cat"])])

    df = try_models(models, X, y, preprocessor)
    # Ensure output shape is as expected
    assert(df.shape == (3, 1))


def get_error(model, X, y):
    """ Evaluates various regression error metrics
    using the given model on the given dataset. 
​
    Parameters
    ----------
    x : DataFrame
        Training data/ test data.
​    y : Series
        Training label/ test label
    Returns
    -------
    errors : list
        List of regression metrics evaluated on the
        given dataset
    """
    y_predicted = model.predict(X)
    mean_abs_error = mean_absolute_error(y, y_predicted)
    mean_sqrd_error = mean_squared_error(y, y_predicted)
    root_mean_sqrd_error = np.sqrt(mean_squared_error(y, y_predicted))
    r2 = r2_score(y, y_predicted)
    exp_var_score = explained_variance_score(y, y_predicted)
    errors = [mean_abs_error, mean_sqrd_error, root_mean_sqrd_error, r2, exp_var_score]

    return errors

def test_get_error():
    model = LinearRegression()
    # Generate some test data
    X = pd.DataFrame({"num1" : [1, 2, 3, 4, 5],
                    "num2" : [10, 26, 3, 18, 11]})
    y = np.array([10, 2, 13, 10, 11])

    model.fit(X, y)
    errors = get_error(model, X, y)
    # Ensure output dimension is as expected
    assert len(errors) == 5


if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt["--training_data_file_path"], opt["--test_data_file_path"], 
            opt["--results_file_location"])
