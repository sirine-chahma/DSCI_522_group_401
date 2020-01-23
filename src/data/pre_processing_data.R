# author: Karanpal	Singh, Sreejith	Munthikodu, Sirine	Chahma
# date: 2020-01-17
#
" This script splits the data given as an input into training and test data.
It saves the two sets in two seperate csv files, in the file that is given as an input.
This script takes two arguments : a path/filename pointing to the data to be read in
and a path/filename pointing to where the cleaned data should live. 

Usage: pre_processing_data.R --input_file=<input_file> --output_dir=<out_dir>

Options:
--input_file=<input_file>             Path (including filename) to raw data (csv file)
--output_dir=<output_dir>   Path to directory where the processed data should be written
" -> doc

library(tidyverse)
library(testthat)
library(docopt)
set.seed(123)

opt <- docopt(doc)
main <- function(input_file, output_dir){

  # read data
  data <- read_csv(input_file)  
  
  # split into training and test data sets
  train_test_data <- train_test_split(data, 0.7)
  train_data <- train_test_data[[1]]
  test_data <- train_test_data[[2]]
  
  # write training and test data to csv files
  write_csv(train_data, paste0(output_dir, "/medical_cost_data_train.csv"))
  write_csv(test_data, paste0(output_dir, "/medical_cost_data_test.csv"))
}


#' splits a dataframe into training and test data sets
#'
#' @param data the dataframe that we want to split
#' @param rate a float between 0 and 1 that represents the proportion of the train data
#' @return a list with the train data as a first value, and the test data as a second value
#' @examples
#' train_data <- train_test_split(data, 0.8)[[1]]
train_test_split <- function(data, rate) {
  sample_size <- floor(rate * nrow(data))
  #take a sample of the numbers of the row that we want to be in our train data set
  train_index <- sample(seq_len(nrow(data)), size = sample_size)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  list(train_data, test_data)
}

#test that the train_test_split function works
test_split <- function(){
  data <- data.frame(c(seq_len(10)), c(seq_len(10)))
  test_that("the length of the train data should be 8 if the initial data as 10 values and the rate is 0.8", {
    expect_equal(nrow(train_test_split(data, 0.8)[[1]]), 8)
    })
  }

test_split()



main(opt[["--input_file"]], opt[["--output_dir"]])