
# Medical Expense Data Analysis and Predictive Modeling

  - author: Karanpal Singh, Sreejith Munthikodu, Sirine Chahma

## About

Knowing the estimated medical expenses in advance is very critical for
insurance companies to set their yearly premiums to beneficiaries. They
should make sure that the amount they spent on beneficiaries is less
than the total premium they receive. Using a predictive model that
predicts the expected medical expense of a customer, the insurance
companies get an estimate of how much premium to charge on each segment
of customers such that the cost results in reasonable profit to the
company. Using such predictive models, they can identify potentially
high-risk customers and charge a higher premium from such customers. In
this project, we attempt to build a regression model that will help us
predict the medical expenses of a person given information such as age,
sex, number of children, smoking habits, and the region where the person
is from. In the process, we are also interested in figuring out if there
is a significant difference in expenses between smokers and non-smokers,
and between males and females. Our hypothesis tests suggest that at 0.05
significance level there is significant evidence to conclude that
smokers incur more medical expenses than non-smokers. Also, there is
statistically no significant evidence to conclude that the medical
expenses of men and women are different. For our predictive model, after
trying different regression models, we found that the decision tree
regressor performs the best on our data. Using our final tuned model, we
achieved satisfying results on the test data set, with a
![R^2](https://latex.codecogs.com/png.latex?R%5E2 "R^2") score of 0.826.

The Data we are using for this analysis is used in the book Machine
Learning with R by Brett Lantz(Lantz 2013); which is a book that
provides an introduction to machine learning using R. All of these
datasets are in the public domain. The data explain the cost of a small
sample of USA population Medical Insurance Cost based on attributes like
age, sex, number of children etc. Additional information about this data
can be found
[here](https://gist.github.com/meperezcuello/82a9f1c1c473d6585e750ad2e3c05a41).

## Report

  - The final report can be found
    [here](https://github.com/UBC-MDS/DSCI_522_group_401/blob/master/reports/medical_expense_analysis.md).
  - Exploratory Data Analysis can be found
    [here](https://github.com/UBC-MDS/DSCI_522_group_401/blob/master/notebooks/EDA.ipynb).

## Usage

There are two suggested ways to run this analysis:

#### 1\. Using Docker

*note - the instructions in this section also depends on running this in
a unix shell (e.g., terminal or Git Bash)*

To replicate the analysis, install
[Docker](https://www.docker.com/get-started). Then clone this GitHub
repository and run the following command at the command line/terminal
from the root directory of this project:

    docker run --rm -v "$(pwd):/home/rstudio/" singhkaranpal/milestone_4 make -C /home/rstudio/ all

To reset the repo to a clean state, with no intermediate or results
files, run the following command at the command line/terminal from the
root directory of this project:

    docker run --rm -v "$(pwd):/home/rstudio/" singhkaranpal/milestone_4 make -C /home/rstudio/ clean

#### 2\. Without using Docker

To replicate the analysis, clone this GitHub repository, install the
[dependencies](#dependencies) listed below, and run the following
command at the command line/terminal from the root directory of this
project:

    make all

To reset the repo to a clean state (which means with no intermediate or
results files), run the following command at the command line/terminal
from the root directory of this project:

    make clean

## Dependency diagram of the Makefile

![Makefile dependencies](reports/Makefile.png)

## Dependencies

  - Python 3.7.3 and Python packages:
      - docopt==0.6.2
      - pandas==0.24.2
      - feather-format==0.4.0
      - scikit-learn==0.22.1
      - altair==4.0.0
      - scipy==1.2.3
      - matplotlib==3.2.1
      - selenium –3.141.0
  - R version 3.6.1 and R packages:
      - knitr==1.26
      - testthat==2.3.1
      - broom==0.5.3
      - feather==0.3.5
      - tidyverse==1.2.1
      - caret==6.0-84
      - ggridges==0.5.1
      - ggthemes==4.2.0
      - selenium 3.141.0
  - ChromeDriver 79.0.3945.36

# References

<div id="refs" class="references hanging-indent">

<div id="ref-source">

Lantz, Brett. 2013. *Machine Learning with R*. PACKT Publishing.

</div>

</div>
