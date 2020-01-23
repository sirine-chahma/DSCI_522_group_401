# author: Karanpal	Singh, Sreejith	Munthikodu, Sirine	Chahma
# date: 2020-01-17
#
" This script cleans the data that can be found in the input path. It does all the 
pre-processing work, it splits the data between a train set and a test set.
It saves the two sets in the data folder in the project directory.
This script takes two arguments : a path/filename pointing to the data to be read in
and a path/filename pointing to where the cleaned data should live. 

Usage: pre_processing_data.R --row_data_loc=<row_data_loc> --clean_data_loc=<clean_data_loc>
" -> doc

library(tidyverse)
library(testthat)
library(docopt)

