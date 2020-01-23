# author: Karanpal	Singh, Sreejith	Munthikodu, Sirine	Chahma
# date: 2020-01-17
#
" This script splits the data given as an input into training andd test data.
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
  sample_size <- floor(0.80 * nrow(data))

  train_index <- sample(seq_len(nrow(data)), size = sample_size)

  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  
  # write training and test data to csv files
  write_csv(train_data, paste0(output_dir, "/medical_cost_data_train.csv"))
  write_csv(test_data, paste0(output_dir, "/medical_cost_data_test.csv"))
}

main(opt[["--input_file"]], opt[["--output_dir"]])