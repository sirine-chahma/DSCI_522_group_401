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

 > For our main question, we will be running several regression models including linear regression, decision tree and random forest. We will pick the best model for prediciting the medical expenses.
 >
 > For the sub-questions, we will be running two two-sided hypothesis tests, comparing the means of the different groups under study. 

### Exploratory Data Analysis

> - In order to know if medical expenses increases with age, and also its bifurcation on sex, we are going to plot the expensive VS the age for each sex.
> - In order to know if medical expenses increases with age, and also its bifurcation on smokers and non-smokers, we are going to plot the expensive VS the age for smokers and non-smokers.

### Results of Analysis
Suggest how you would share the results of your analysis as one or more tables and/or figures.

> - R-squared, ajusted R-squared values or any relevant metric
> - Best fit line plot (depending on the features that will be significant)
> - p-values for inferential sub-questions

### Usage

```{}
#split the data
Rscript src/data/pre_processing_data.R --input_file=data/original/medical_cost_data.csv --output_dir=data/processed
``` 