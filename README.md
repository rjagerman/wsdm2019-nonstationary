# Off-policy Evaluation in Non-stationary Recommendation Environments
This repository contains the code to reproduce the experiments for the WSDM2019 paper "When People Change Their Mind: Off-Policy Evaluation in Non-stationary Recommendation Environments".

## Prerequisites
The code depends on a working installation of `pipenv` (can be installed from ) and `make` (should be on your machine if you run some Unix-variant or OSX). To install the python dependencies, run the following from the root of the git repository:

    pipenv install

## Running the code
The entire experimental pipeline is automated via a Makefile. From the root of the git repository you can simply run:

    pipenv run make

If you have more than 1 CPU core, you can parallelize parts of the pipeline by running the following (e.g. if you have 4 cores):

    pipenv run make -j4

The make process will go through all the steps necessary to reproduce the experiments: downloading the data, training candidate policies and running our estimators. Even with 60 cores the entire process will takes a few days to complete. The majority of time is spend running the estimator simulations.

## Output
Output is stored in the folder 'build/'. This includes the processed datasets, trained policies and result files.

