Medical Expense Data Analysis and Predictive Modeling
================

# Summary of the data set

  - summary of the data  
  - prject proposal
  - research quetions

# Exploratory analysis on the training data set

To understand the nature of predictors with respect to `Medical
Expenses` we will perform Exploratory Data Analysis and we will try to
understand if there are some intresting behaviours.

##### 1\. Let’s see how `Medical Expenses` are changing with `Age`

<html>

<img src = '../../reports/figures/1.Expenses_VS_Age.png'>

</html>

It can be observed that `Medical Expense` of people is increasing, as
`Age`
increases.

##### 2\. Let’s see how `Medical Expenses` are changing with `BMI (Body Mass Index)`

<html>

<img src = '../../reports/figures/2.Expenses_VS_BMI.png'>

</html>

The highest expenses seem to occur for people who have a higher
BMI.

##### 3\. Let’s see how much money males and females spending on medical treatments between 18-64 Years

<html>

<img src = '../../reports/figures/3.Expenses_VS_Gender.png'>

</html>

The expenses seem to grow with age for both males and females. It looks
like Males in their **20’s & 60’s** tend to pay more on their `Medical
Expenses` than Females. Females in their **40’s** are paying more than
Males on their `Medical
Expenses`.

##### 4\. Let’s see how `Smokers` and `Non-Smokers` are spending on medical treatments between 18-64 Years

We expect expenditures by smokers should be higher than the non smokers.

<html>

<img src = '../../reports/figures/4.Expenses_VS_Smoker.png'>

</html>

**Interesting\!\!\!** - As expected, Health expenses of smokers are a
lot higher than the one of non-smokers.

##### 5\. Let’s see how `BMI` is changing with Age for Males and Females

We are expecting both male and females have usual `BMI`.

<html>

<img src = '../../reports/figures/5.BMI_VS_AGE.png'>

</html>

The `BMI` doesn’t seem to vary depending on the age nor the sex.

##### 6\. Let’s see the Male & Female Expenses Over BMI

<html>

<img src = '../../reports/figures/6.EXP_VS_BMI.png'>

</html>

The highest expenses seem to occur for people from both genders who have
a BMI that is higher than 34.

# Answering the Inferential Research Questions

Now, from above Exploratory Data Analysis we are interested in following
two questions:

  - Is there a significant difference of expenses between smokers and
    non-smokers?
  - Is there a significant difference of expenses between males and
    females?

<br>

##### 1\. Is there a significant difference of expenses between smokers and non-smokers?

  
![H\_0: \\mu\_{Smokers} =
\\mu\_{Non-Smokers}](https://latex.codecogs.com/png.latex?H_0%3A%20%5Cmu_%7BSmokers%7D%20%3D%20%5Cmu_%7BNon-Smokers%7D
"H_0: \\mu_{Smokers} = \\mu_{Non-Smokers}")  
  
![H\_1: \\mu\_{Smokers} \\neq
\\mu\_{Non-Smokers}](https://latex.codecogs.com/png.latex?H_1%3A%20%5Cmu_%7BSmokers%7D%20%5Cneq%20%5Cmu_%7BNon-Smokers%7D
"H_1: \\mu_{Smokers} \\neq \\mu_{Non-Smokers}")  

Our Null hypothesis states that mean expenses of smokers is equal to
mean expenses of non-smokers and Alternate hypothesis states that there
is a significant difference between these two quantities. We have used
t-test to compare mean of two groups and test results are as following:

<table>

<thead>

<tr>

<th style="text-align:right;">

estimate

</th>

<th style="text-align:right;">

estimate1

</th>

<th style="text-align:right;">

estimate2

</th>

<th style="text-align:right;">

statistic

</th>

<th style="text-align:right;">

p.value

</th>

<th style="text-align:right;">

parameter

</th>

<th style="text-align:right;">

conf.low

</th>

<th style="text-align:right;">

conf.high

</th>

<th style="text-align:left;">

method

</th>

<th style="text-align:left;">

alternative

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:right;">

\-23615.96

</td>

<td style="text-align:right;">

8434.268

</td>

<td style="text-align:right;">

32050.23

</td>

<td style="text-align:right;">

\-32.75189

</td>

<td style="text-align:right;">

0

</td>

<td style="text-align:right;">

311.8511

</td>

<td style="text-align:right;">

\-25034.71

</td>

<td style="text-align:right;">

\-22197.21

</td>

<td style="text-align:left;">

Welch Two Sample t-test

</td>

<td style="text-align:left;">

two.sided

</td>

</tr>

</tbody>

</table>

The exact p-value is
![5.8894644\\times 10^{-103}](https://latex.codecogs.com/png.latex?5.8894644%5Ctimes%2010%5E%7B-103%7D
"5.8894644\\times 10^{-103}") which is very close to 0. However, while
rendering the output to the table above it treats it as 0. As, we can
observe that the p-values is less than the significance level of
![5\\%](https://latex.codecogs.com/png.latex?5%5C%25 "5\\%") hence, we
can reject ![H\_0](https://latex.codecogs.com/png.latex?H_0 "H_0")
hypothesis and conclude that we didn’t have enough evidence to say mean
expenses between smoker and no-smokers are
same.

##### 2\. Is there a significant difference of expenses between males and females?

  
![H\_0: \\mu\_{Males} =
\\mu\_{Females}](https://latex.codecogs.com/png.latex?H_0%3A%20%5Cmu_%7BMales%7D%20%3D%20%5Cmu_%7BFemales%7D
"H_0: \\mu_{Males} = \\mu_{Females}")  
  
![H\_1: \\mu\_{Males} \\neq
\\mu\_{Females}](https://latex.codecogs.com/png.latex?H_1%3A%20%5Cmu_%7BMales%7D%20%5Cneq%20%5Cmu_%7BFemales%7D
"H_1: \\mu_{Males} \\neq \\mu_{Females}")  

Our Null hypothesis states that mean expenses of males is equal to mean
expenses of females and Alternate hypothesis states that there is a
significant difference between these two quantities. We have used t-test
to compare mean of two groups and test results are as following:

<table>

<thead>

<tr>

<th style="text-align:right;">

estimate

</th>

<th style="text-align:right;">

estimate1

</th>

<th style="text-align:right;">

estimate2

</th>

<th style="text-align:right;">

statistic

</th>

<th style="text-align:right;">

p.value

</th>

<th style="text-align:right;">

parameter

</th>

<th style="text-align:right;">

conf.low

</th>

<th style="text-align:right;">

conf.high

</th>

<th style="text-align:left;">

method

</th>

<th style="text-align:left;">

alternative

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:right;">

\-1387.172

</td>

<td style="text-align:right;">

12569.58

</td>

<td style="text-align:right;">

13956.75

</td>

<td style="text-align:right;">

\-2.100888

</td>

<td style="text-align:right;">

0.035841

</td>

<td style="text-align:right;">

1313.36

</td>

<td style="text-align:right;">

\-2682.489

</td>

<td style="text-align:right;">

\-91.85535

</td>

<td style="text-align:left;">

Welch Two Sample t-test

</td>

<td style="text-align:left;">

two.sided

</td>

</tr>

</tbody>

</table>

The exact p-value is
![0.035841](https://latex.codecogs.com/png.latex?0.035841 "0.035841")
which is less than the significance level of
![5\\%](https://latex.codecogs.com/png.latex?5%5C%25 "5\\%") hence, we
can reject ![H\_0](https://latex.codecogs.com/png.latex?H_0 "H_0")
hypothesis and conclude that we didn’t have enough evidence to say mean
expenses between Males and Females are same.

# Build a predictive model

  - description
  - model selection
  - hyper-parameter tuning

# Evaluate the predictive model

  - model evaluation on train and test
  - results
  - regression plots

# References
