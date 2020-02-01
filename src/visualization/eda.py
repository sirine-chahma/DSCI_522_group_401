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
    
    # dataset can't be empty
    assert X_train.shape != (0, 0), 'Empty Dataset: X'
    assert len(y_train != 0), 'Empty Dataset: y'

    training_df = X_train
    training_df['charges'] = y_train
    
    #plotting the correlation between features
    cor = X_train.corr()
    cor = cor.reset_index()
    cor = pd.melt(cor, id_vars="index")
    cor_map = alt.Chart(cor).mark_rect().encode(
    alt.X('index:O', title=''),
    alt.Y('variable:O', title=''),
    alt.Color('value:Q',scale = alt.Scale(domain=[0, 1], scheme = 'purplered'))
    ).properties(title="Correlation map of the features", width = 700, height = 300
    ).configure_axis(labelFontSize = 15, titleFontSize = 15
    ).configure_title(fontSize = 15)
    
    # saving results as png
    cor_map.save(output_location + '/0.correlation.png')

    # plotting relationship between Medical Expenses over Age
    meoa = alt.Chart(training_df.groupby(['age']).mean().reset_index()).mark_line(point = True).encode(
        alt.X('age:N', title = 'Age'),
        alt.Y('charges:Q', title = 'Expenses')
    ).properties(title='Medical Expenses over Age', width = 700, height = 300
    ).configure_axis(labelFontSize = 15, titleFontSize = 15
    ).configure_title(fontSize = 15)
    
    # saving results as png
    meoa.save(output_location + '/1.Expenses_VS_Age.png')

    # plotting relationship between Medical Expenses vs BMI
    
    training_df = X_train
    training_df['charges'] = y_train

    training_df['bmi_cat'] = 'normal'
    training_df.loc[training_df['bmi'] < 18.5,'bmi_cat'] = 'underweigth'
    training_df.loc[training_df['bmi'] > 25,'bmi_cat'] = 'overweigth'
    training_df.loc[training_df['bmi'] > 30,'bmi_cat'] = 'obese'


    bmi = alt.Chart(training_df.groupby(['bmi_cat']).mean().reset_index()).mark_bar().encode(
    alt.X('bmi_cat:N', title = 'BMI', sort=['underweigth', 'normal', 'overweight', 'obese']),
    alt.Y('charges:Q', title = 'Expenses')
    ).properties(title='Medical Expenses vs BMI', width = 700, height = 300
    ).configure_axis(labelFontSize = 20, titleFontSize = 20
    ).configure_title(fontSize = 20)

    # saving results as png
    bmi.save(output_location + '/2.Expenses_VS_BMI.png')

    # plotting relationship between Male and Female Medical Expenses Over Age
    exp_gender = alt.Chart(training_df.groupby(['age','sex']).mean().reset_index()).mark_line(point = True).encode(
        alt.X('age:N', title = 'Age'),
        alt.Y('charges:Q', title = 'Expenses'),
        alt.Color('sex:N')
    ).properties(title='Male and Female Medical Expenses Over Age', width = 700, height = 300
    ).configure_axis(labelFontSize = 15, titleFontSize = 15
    ).configure_title(fontSize = 15)

    # saving results as png
    exp_gender.save(output_location + '/3.Expenses_VS_Gender.png')

    # plotting relationship between Smokers & Non Smokers Medical Expenses Over Age
    exp_smoke = alt.Chart(training_df.groupby(['age','smoker']).mean().reset_index()).mark_line(point = True).encode(
        alt.X('age:N', title = 'Age'),
        alt.Y('charges:Q', title = 'Expenses'),
        alt.Color('smoker:N')
    ).properties(title='Smokers & Non Smokers Medical Expenses Over Age', width = 700, height = 300
    ).configure_axis(labelFontSize = 15, titleFontSize = 15
    ).configure_title(fontSize = 15)

    # saving results as png
    exp_smoke.save(output_location + '/4.Expenses_VS_Smoker.png')


    # plotting relationship between Male & Female Expenses Over BMI
    exp_bmi = alt.Chart(training_df.groupby(['bmi_cat','sex']).mean().reset_index()).mark_bar().encode(
        alt.X('bmi_cat:O', title = 'BMI', sort=['underweigth', 'normal', 'overweight', 'obese']),
        alt.Y('charges:Q', title = 'Expenses'),
        alt.Color('sex:N')
    ).properties(title='Male & Female Expenses Over BMI', width = 700, height = 300
    ).configure_axis(labelFontSize = 20, titleFontSize = 20
    ).configure_title(fontSize = 20)

    # saving results as png
    exp_bmi.save(output_location + '/6.EXP_VS_BMI.png')


if __name__ == "__main__":
    main(opt["--input_data"], opt["--output_location"])




