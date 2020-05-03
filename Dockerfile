# Docker file for the Medical Analysis & Prediction Modelling
# author: Karanpal Singh, Sreejith Munthikodu & Sirine Chahma
# date: Feb 06, 2020

# use rocker/tidyverse as the base image and
FROM rocker/tidyverse

# install R packages
RUN apt-get update -qq && apt-get -y --no-install-recommends install \
  && install2.r --error \
    --deps TRUE \
    cowsay \
    here \
    feather \
    ggridges \
    ggthemes \
    e1071 \
    caret 

# install the R packages using install.packages
RUN Rscript -e "install.packages('kableExtra')"
RUN Rscript -e "install.packages('testthat')"
RUN Rscript -e "install.packages('docopt')"
RUN Rscript -e "install.packages('tidyverse')"
RUN Rscript -e "install.packages('broom')"
RUN Rscript -e "install.packages('kable')"

# install the anaconda distribution of python
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    /opt/conda/bin/conda update -n base -c defaults conda

# put anaconda python in path
ENV PATH="/opt/conda/bin:${PATH}"

# install docopt python package
RUN conda install -y docopt requests pandas numpy matplotlib seaborn altair scipy
  
# Packages for Altair to Save plots  
RUN conda install -y selenium
RUN apt-get install chromium -y

# install chromeDriver
RUN LATEST=$(wget -q -O - http://chromedriver.storage.googleapis.com/LATEST_RELEASE)
RUN apt-get install -y unzip
RUN wget http://chromedriver.storage.googleapis.com/2.41/chromedriver_linux64.zip
RUN unzip chromedriver_linux64.zip && ln -s $PWD/chromedriver /usr/local/bin/chromedriver
RUN chromedriver -v