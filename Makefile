# author: Karanpal Singh, Sreejith Munthikodu, Sirine Chahma
# date: 2020-01-31

all: reports/medical_expense_analysis.md

# download data
#
# This script will download the data from URL and save it in project's data directory.
# Sreejith is author of this file and created in Milestone 1
# We have included this in Makefile in Milestone 3
data/original/medical_cost_data.csv : src/processing_data/get_data.py
	python src/processing_data/get_data.py --url=https://gist.githubusercontent.com/meperezcuello/82a9f1c1c473d6585e750ad2e3c05a41/raw/d42d226d0dd64e7f5395a0eec1b9190a10edbc03/Medical_Cost.csv --file_location=data/original/medical_cost_data.csv
	
# Split the data
#
# This script will pre-process downloaded data into training and test data files.
# Sirine is author of this file and created this in Milestone 1
# We have included this in Makefile in Milestone 3
data/processed/medical_cost_data_train.csv data/processed/medical_cost_data_test.csv : src/processing_data/pre_processing_data.R data/original/medical_cost_data.csv
	Rscript src/processing_data/pre_processing_data.R --input_file=data/original/medical_cost_data.csv --output_dir=data/processed
	
# EDA Script
# 
# This script will run Exploratory Data Analysis on Pre-processed Data
# and it will save EDA plots as images in local repo for report creation
# 
# Karan is author of this file and create this in Milestone 2
# We have included this script in Makefile in Milestone3
reports/figures/0.correlation.png reports/figures/1.Expenses_VS_Age.png reports/figures/2.Expenses_VS_BMI.png reports/figures/3.Expenses_VS_Gender.png reports/figures/4.Expenses_VS_Smoker.png reports/figures/6.EXP_VS_BMI.png : src/visualization/eda.py data/processed/medical_cost_data_train.csv data/processed/medical_cost_data_test.csv data/original/medical_cost_data.csv
	python src/visualization/eda.py --input_data=data/processed/medical_cost_data_train.csv --output_location=reports/figures

# Inferences
#
# This script will run Inferential Analysis and perform hypothesis testing
# outputs are used in report creation
#
# Karan is author of this file and create this in Milestone 2
# We have included this script in Makefile in Milestone3
reports/tables/1.hypothesis_smokers.csv reports/tables/2.hypothesis_sex.csv : src/inferences/inferences.R data/processed/medical_cost_data_train.csv data/processed/medical_cost_data_test.csv data/original/medical_cost_data.csv
	Rscript src/inferences/inferences.R --input_file=data/original/medical_cost_data.csv --output_dir=reports/tables/
	
# Predictive Modelling
#
# This script will run predictive modelling on processed data and
# will compare few model performances
#
# Sreejith is author of this file and created in Milestone 2
# We have included this in Makefile in Milestone 3 
reports/tables/preprocessors.csv reports/tables/regression_models_base_errors.csv reports/tables/hyperparameters.csv reports/tables/regression_errors.csv reports/figures/predicted_vs_actual_plot.png reports/figures/residual_plot.png : src/models/train_predict_medical_expense.py data/processed/medical_cost_data_train.csv data/processed/medical_cost_data_test.csv data/original/medical_cost_data.csv
	python src/models/train_predict_medical_expense.py --training_data_file_path="data/processed/medical_cost_data_train.csv" --test_data_file_path="data/processed/medical_cost_data_test.csv" --results_file_location="reports"

# render report
#
# This script will create a project report and render it as
# md and html.
#
# Karan and Sirine are authors of this script and this was created in Milestone 2
# We have included this in Makefile in Milestone 3
reports/medical_expense_analysis.md : reports/medical_expense_analysis.Rmd docs/medical_expense_refs.bib reports/tables/preprocessors.csv reports/tables/regression_models_base_errors.csv reports/tables/hyperparameters.csv reports/tables/regression_errors.csv reports/figures/predicted_vs_actual_plot.png reports/figures/residual_plot.png reports/tables/1.hypothesis_smokers.csv reports/tables/2.hypothesis_sex.csv reports/figures/0.correlation.png reports/figures/1.Expenses_VS_Age.png reports/figures/2.Expenses_VS_BMI.png reports/figures/3.Expenses_VS_Gender.png reports/figures/4.Expenses_VS_Smoker.png reports/figures/6.EXP_VS_BMI.png data/processed/medical_cost_data_train.csv data/processed/medical_cost_data_test.csv data/original/medical_cost_data.csv
	Rscript -e "library(rmarkdown);render('reports/medical_expense_analysis.Rmd')"

# cleaning everything
clean:
	rm -rf data/original/*
	rm -rf data/processed/*
	rm -rf reports/figures/*.png
	rm -rf reports/tables/*
	rm -rf reports/*.md
	rm -rf reports/*.html