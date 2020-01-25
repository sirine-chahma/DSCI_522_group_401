
# Medical Expense Data Analysis and Predictive Modeling

  - author: Karanpal Singh, Sreejith Munthikodu, Sirine Chahma

## About

In this project, we attemps to build a regression model that will help
us predict the medical expenses of a person regarding some information
about this person (age, sex, number of children, if the person smokes
and the region where the person is from). After trying different types
of regressors (linear regression, decision tree regressor, knn
regression, random forest regression and SVR), we found out that the
decision tree regressor ended up being the best model regarding to our
data. Our final regressor had satisfying results on an unseen data set,
with a ![R^2](https://latex.codecogs.com/png.latex?R%5E2 "R^2") score of
0.894 on our test data set.

This project will attempt to build a predictive model that will answer
the following question : “Given a person’s information, what would be
his/her predicted medical expenses?”. Ansewering this question can be
important for insurance compagnies who wants to evaluate the risk to
insure a certain person regarding to his/her possible medical expenses.

We also wanted to figure out if there is a significant difference of
expenses between smokers and non-smokers, and between males and females.
Therefore, we led two inferential studies asside in order to find an
aswer to those questions.

The Data we are using for this analysis is used in the book Machine
Learning with R by Brett Lantz(Lantz 2013); which is a book that
provides an introduction to machine learning using R. All of these
datasets are in the public domain. The data explain the cost of a small
sample of USA population Medical Insurance Cost based on attributes like
age, sex, number of children etc. Additional information about this data
can be found
[here](https://gist.github.com/meperezcuello/82a9f1c1c473d6585e750ad2e3c05a41).

### Research Question

For this project we are working on the following main predictive
research question

Our research question is “Given a person’s information, what would be
his/her predicted medical expenses?”

Throughout this project we also want to investigate two more inferential
sub-questions

  - Is there a significant difference of expenses between smokers and
    non-smokers? (inferential)
  - Is there a significant difference of expenses between males and
    females? (inferential)

### Analysis Plan

For our predictive modeling, the analysis plan is summarised below :

  - Transform numeric features using PolynomialFeatures and
    StandardScaler in sci-kit learn.  
  - Transform categorical features using OneHotEncoder in sci-kit
    learn.  
  - Get base performance using the following regression models.
    1.  Linear Regression  
    2.  KNN Regressor  
    3.  DecisionTreeRegressor  
    4.  RandomForestRegressor  
    5.  SVR(SupportVectorRegressor)  
  - Choose a model from above based on performance  
  - Perform hyper-parameter tuning  
  - Evaluate the model using the following metrics
    1.  Mean absolute error  
    2.  Mean squared error  
    3.  Root mean squared error  
    4.  R2 Score  
    5.  Explained varoance score  
  - Plot predicted expense vs actual expense plot and residual plot

For the sub-questions, we will be running two two-sided hypothesis
tests, comparing the means of the different groups under study.

### Exploratory Data Analysis

  - In order to know if medical expenses increases with age, and also
    its bifurcation on sex, we are going to plot the expensive VS the
    age for each sex.

  - In order to know if medical expenses increases with age, and also
    its bifurcation on smokers and non-smokers, we are going to plot the
    expensive VS the age for smokers and non-smokers.

### Results of Analysis

Suggest how you would share the results of your analysis as one or more
tables and/or figures.

  - Evaluation metrics
    1.  Mean absolute error  
    2.  Mean squared error  
    3.  Root mean squared error  
    4.  R2 Score  
    5.  Explained varoance score  
  - Best fit line plot
    1.  Predicted expense vs Actual expense plot and residual plot  
  - p-values for inferential sub-questions

## Report

The final report can be found
[here](https://github.com/UBC-MDS/DSCI_522_group_401/blob/master/src/reports/medical_expense_analysis.md).

## Usage

To replicate the analysis, clone this GitHub repository, install the
[dependencies](#dependencies) listed below, and run the following
command at the command line/terminal from the root directory of this
project:

    # Split the data
    Rscript src/data/pre_processing_data.R --input_file=data/original/medical_cost_data.csv --output_dir=data/processed

    # EDA Script
    python src/visualization/eda.py --input_data=data/processed/medical_cost_data_train.csv --output_location=reports/figures

    # Inferences
    Rscript src/inferences/inferences.R --input_file=data/original/medical_cost_data.csv --output_dir=reports/tables/

    # Predictive Modeling
    python src/models/train_predict_medical_expense.py --training_data_file_path="data/processed/medical_cost_data_train.csv" --test_data_file_path="data/processed/medical_cost_data_test.csv" --results_file_location="reports"

## Dependencies

  - Python 3.7.3 and Python packages:
      - docopt==0.6.2
      - pandas==0.24.2
      - feather-format==0.4.0
      - scikit-learn==0.22.1
      - altair==4.0.0
      - scipy==1.2.3
      - matplotlib==3.2.1
  - R version 3.6.1 and R packages:
      - knitr==1.26
      - testthat==2.3.1
      - broom==0.5.3
      - feather==0.3.5
      - tidyverse==1.2.1
      - caret==6.0-84
      - ggridges==0.5.1
      - ggthemes==4.2.0

# References

<div id="refs" class="references">

<div id="ref-source">

Lantz, Brett. 2013. *Machine Learning with R*. PACKT Publishing.

</div>

</div>
