# author: Karanpal Singh, Sreejith Munthikodu, Sirine Chahma
# date: 2020-01-22

'''
This script will run exploratory data analysis on downloaded and
cleaned data in data folder. This script will take two arguments,
an input path to data and an output file location for saving EDA plots.

Usage: eda.py --input_data=<input_data> --output_location=<output_location>
 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from selenium import webdriver
from sklearn.model_selection import train_test_split
from docopt import docopt

opt = docopt(__doc__)

def main(input_data, output_location):
    # load training data
    medical_data = pd.read_csv(input_data)
    X_train = medical_data.drop(columns="charges")
    y_train = medical_data.charges


    training_df = X_train
    training_df['charges'] = y_train


    meoa = alt.Chart(training_df.groupby(['age']).mean().reset_index()).mark_line(point = True).encode(
        alt.X('age:N', title = 'Age'),
        alt.Y('charges:Q', title = 'Expenses')
    ).properties(title='Medical Expenses over Age')

    meoa.save(output_location + '/1.Expenses_VS_Age.png')


    bmi = alt.Chart(training_df.groupby(['bmi']).mean().reset_index()).mark_area().encode(
        alt.X('bmi:Q', title = 'BMI', bin=alt.Bin(maxbins=60)),
        alt.Y('charges:Q', title = 'Expenses')
    ).properties(title='Medical Expenses vs BMI')


    bmi.save(output_location + '/2.Expenses_VS_BMI.png')


    exp_gender = alt.Chart(training_df.groupby(['age','sex']).mean().reset_index()).mark_line(point = True).encode(
        alt.X('age:N', title = 'Age'),
        alt.Y('charges:Q', title = 'Expenses'),
        alt.Color('sex:N')
    ).properties(title='Male and Female Medical Expenses Over Age')


    exp_gender.save(output_location + '/3.Expenses_VS_Gender.png')


    exp_smoke = alt.Chart(training_df.groupby(['age','smoker']).mean().reset_index()).mark_line(point = True).encode(
        alt.X('age:N', title = 'Age'),
        alt.Y('charges:Q', title = 'Expenses'),
        alt.Color('smoker:N')
    ).properties(title='Smokers & Non Smokers Medical Expenses Over Age')


    exp_smoke.save(output_location + '/4.Expenses_VS_Smoker.png')


    bmi_age = alt.Chart(X_train).mark_point().encode(
        
        alt.X('age:N', title = 'Age'),
        alt.Y('bmi:Q', title = 'BMI'),
        alt.Color('sex:N')
    ).properties(title='Males & Females BMI Over Age')

    bmi_age.save(output_location + '/5.BMI_VS_AGE.png')


    exp_bmi = alt.Chart(training_df.groupby(['bmi','sex']).mean().reset_index()).mark_area().encode(
        alt.X('bmi:Q', title = 'BMI', bin=alt.Bin(maxbins=60)),
        alt.Y('charges:Q', title = 'Expenses'),
        alt.Color('sex:N')
    ).properties(title='Male & Female Expenses Over BMI')


    exp_bmi.save(output_location + '/6.EXP_VS_BMI.png')


if __name__ == "__main__":
    main(opt["--input_data"], opt["--output_location"])




