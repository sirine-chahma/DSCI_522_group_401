Medical Expense Data Analysis and Predictive Modeling
================

# Summary of the data set

  - summary of the data  
  - prject proposal
  - research quetions

# Exploratory analysis on the training data set

  - description
  - plots
  - observatons

# Answer the inferential research questions

  - probelm statement
  - hypothesis test
  - inference

# Build a predictive model

In this data analysis project, we primarily focus on predicting the
medical expenses given the details of a customer. We used Python for
building the machine learnining model. The machine learning library,
`sci-kit learn` was extensively used in this project to transform the
data, feature engineering, feature selection, model selection,
hyper-parameter tuning and model evaluation.

### Preprocessing

Firtly, the training data was loaded and response variable was separated
from the training data. Then numerical and categorical features in the
data are identified. A summary of various feature transformations done
on the categorical and numerical features are given below.

| Numeric.features   | Categorical.features |
| :----------------- | :------------------- |
| PolynomialFeatures | OneHotEncoder        |
| StandardScaler     |                      |

After preprocessing and feature transformations, various regression
models are fitted on the training data with the default parameters.
Model with the best performance on the training and validation dataset
is selected for hyper-parameter optimization. A summary of baseline
performance by various regression models are givem
below.

| Error Metrics                   | Linear.Regression | Decision.Tree.Regressor | KNN.Regressor | Random.Forest.Regressor | SVR..Support.Vector.Regressor. |
| :------------------------------ | ----------------: | ----------------------: | ------------: | ----------------------: | -----------------------------: |
| Mean absolute error, training   |         4062.5430 |                  0.0000 |     2739.4530 |                943.1160 |                       8295.865 |
| Mean absolute error, validation |         4102.0070 |               2887.3280 |     3654.2210 |               2463.0710 |                       8313.009 |
| training\_time(s)               |            0.0646 |                  0.1348 |        0.0765 |                  1.2972 |                          0.248 |

Based on the above scores, DecisionTreeRegressor was selected as the
final model and hyper-parameter tuning is done on it. In the data
analysis pipeline, selection of the model from the base models is
currently done manually.

### hyper-parameter tuning

Hyper-parameter optimiation was performed using `GridSearchCV`. The best
parameters obtained from hyper-parameter optimization are given below.

| Hyper-parameter                       | Best.parameter |
| :------------------------------------ | -------------: |
| preprocessor\_\_num\_\_poly\_\_degree |              2 |
| regressor\_\_max\_depth               |              4 |
| regressor\_\_min\_samples\_split      |              4 |

# Evaluate the predictive model

### model evaluation on train and test

The final tuned DecisionTreeRegressor model was evaluated on both the
training and test data using various regression metrics. A summary of
the results are shown below.

| Evaluation Metric          | training.data |    test.data |
| :------------------------- | ------------: | -----------: |
| Mean Absolute Error        |  2.336006e+03 | 2.780371e+03 |
| Mean Squared Error         |  1.739013e+07 | 2.506713e+07 |
| Root Mean Squared Error    |  4.170148e+03 | 5.006708e+03 |
| R2\_score                  |  8.821358e-01 | 8.261748e-01 |
| Explained\_variance\_score |  8.821358e-01 | 8.264199e-01 |

A mean absolute error of 2780.3705362 is not a very good score for the
regression model. However, considering the mean medical expense of
1.322343110^{4}, we are not very far from predicting the accurate
expenses. The poor performance of the model could be because of lack of
enough data, lack of relevant features or the model is not tuned
completely. Considering the limited time available for the project, we
have not done thorough feature engineering, feature selection, model
selection and hyper-parameter tuning. But this serves as a very good
base model on which further improvements can be made. The goodness of
fit of the regression model is analysed in the following
section.

### Goodness of fit

![](../../reports/figures/predicted_vs_actual_plot.png)<!-- -->![](../../reports/figures/residual_plot.png)<!-- -->

From the predicted Vs Actual plot, we can see ther are some errors in
prediction at lower expenses. Overall the model does a pretty decent job
of predicting the medical expenses given the patient information.

# References
