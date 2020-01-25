# author: Karanpal Singh, Sreejith	Munthikodu, Sirine	Chahma
# date: 2020-01-23
#
" This script will perform hypothesis test using t-test
It will read data from user specified location and save outputs to
user specified locations in csv format.

Usage: inferences.R --input_file=<input_file> --output_dir=<out_dir>

Options:
--input_file=<input_file>   Path (including filename) to raw data (csv file)
--output_dir=<output_dir>   Path to directory where the inferences output should be written
" -> doc

library(tidyverse)
library(testthat)
library(docopt)
library(broom)

set.seed(123)

test_smokers <- 0
test_sex <- 0

opt <- docopt(doc)
main <- function(input_file, output_dir){
  
  # read data
  data <- read_csv(input_file)  
  
  # hypothesis test for mean expenses difference between smokers and non-smokers
  test_smokers <- t.test(data$charges ~ data$smoker, mu = 0, alt = 'two.sided', conf = 0.95, var.eq = FALSE, paired = FALSE)
  
  # hypothesis test for mean expenses difference between males and females
  test_sex <- t.test(data$charges ~ data$sex, mu = 0, alt = 'two.sided', conf = 0.95, var.eq = FALSE, paired = FALSE)
  
  # write training and test data to csv files
  write_csv(broom::tidy(test_smokers), paste0(output_dir, "/1.hypothesis_smokers.csv"))
  write_csv(broom::tidy(test_sex), paste0(output_dir, "/2.hypothesis_sex.csv"))
}



#test that the if hypothesis didn't return null
test_split <- function(test_smokers,test_sex){

  test_that("hypothesis results of smokers cannot be null, please check you input", {
    expect_equal(!is.null(test_smokers), TRUE)
  })
  
  test_that("hypothesis results of sex cannot be null, please check you input", {
    expect_equal(!is.null(test_sex), TRUE)
  })
  
}

test_split(test_smokers,test_sex)



main(opt[["--input_file"]], opt[["--output_dir"]])