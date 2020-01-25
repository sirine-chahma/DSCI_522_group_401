# DSCI_522_group_401
## Project Proposal: Medical Expenses Prediction

### Data Source

The [Data](https://gist.githubusercontent.com/meperezcuello/82a9f1c1c473d6585e750ad2e3c05a41/raw/d42d226d0dd64e7f5395a0eec1b9190a10edbc03/Medical_Cost.csv) we are using for this analysis is used in the book Machine Learning with R by Brett Lantz; which is a book that provides an introduction to machine learning using R. All of these datasets are in the public domain. The data explain the cost of a small sample of USA population Medical Insurance Cost based on attributes like age, sex, number of children etc. Additional information about this data can be found [here](https://gist.github.com/meperezcuello/82a9f1c1c473d6585e750ad2e3c05a41).

### Research Question

For this project we are working on the following main predictive research question

> Our research question is "Given a person's information, what would be his/her predicted medical expenses?"

Throughout this project we also want to investigate two more inferential sub-questions

- Is there a significant difference of expenses between smokers and non-smokers? (inferential)
- Is there a significant difference of expenses between males and females? (inferential)


### Analysis Plan

 > For our predictive modeling, the analysis plan is summarised below.  
    - Transform numeric features using PolynomialFeatures and StandardScaler in sci-kit  learn.  
    - Transform categorical features using OneHotEncoder in sci-kit learn.  
    - Get base performance using the following regression models.  
        1. Linear Regression  
        2. KNN Regressor  
        3. DecisionTreeRegressor  
        4. RandomForestRegressor  
        5. SVR(SupportVectorRegressor)  
    - Choose a model from above based on performance  
    - Perform hyper-parameter tuning  
    - Evaluate the model using the following metrics  
        1. Mean absolute error  
        2. Mean squared error  
        3. Root mean squared error  
        4. R2 Score  
        5. Explained varoance score  
    - Plot predicted expense vs actual expense plot and residual plot   
        
 >
 > For the sub-questions, we will be running two two-sided hypothesis tests, comparing the means of the different groups under study. 

### Exploratory Data Analysis

> - In order to know if medical expenses increases with age, and also its bifurcation on sex, we are going to plot the expensive VS the age for each sex.
> - In order to know if medical expenses increases with age, and also its bifurcation on smokers and non-smokers, we are going to plot the expensive VS the age for smokers and non-smokers.

### Results of Analysis
Suggest how you would share the results of your analysis as one or more tables and/or figures.

> - Evaluation metrics  
        1. Mean absolute error    
        2. Mean squared error    
        3. Root mean squared error    
        4. R2 Score    
        5. Explained varoance score    
> - Best fit line plot   
        1. Predicted expense vs Actual expense plot and residual plot    
> - p-values for inferential sub-questions  

### Usage

```{}
# Split the data
Rscript src/data/pre_processing_data.R --input_file=data/original/medical_cost_data.csv --output_dir=data/processed
``` 

```{}
# EDA Script
python eda.py --input_data=../../data/processed/medical_cost_data_train.csv --output_location=../../reports/figures
```

```{}
# Inferences
Rscript src/inferences/inferences.R --input_file=data/original/medical_cost_data.csv --output_dir=reports/tables/
```

```{}
# Predictive Modeling
python train_predict_medical_expense.py --training_data_file_path="../../data/processed/medical_cost_data_train.csv" --test_data_file_path="../../data/processed/medical_cost_data_test.csv" --results_file_location="../../reports"
```