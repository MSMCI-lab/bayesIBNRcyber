# Bayesian IBNR Model for Cyber Incidents

This repository implements a Bayesian model for estimating incurred but not reported (IBNR) cyber incidents. Using the `nimble` package in R, this model leverages a negative binomial likelihood with time- and delay-specific predictors to forecast incident counts based on historical data. This project is designed to provide practitioners and researchers with a framework for IBNR modeling in the context of cyber risk.


## Overview

Cyber incidents often go unreported for a period after they occur, presenting challenges for risk assessment. The Bayesian IBNR model implemented here addresses this by estimating the probability of incidents based on delayed reporting data. This model enables effective prediction of unreported incidents, which can be used for risk assessment and insurance modeling.

## Installation

### Requirements

- R (version >= 4.0.0)
- Required R packages:
  - `nimble`
  - `coda`
  - `doParallel`
 

## Data Preparation

The model expects a data file with incident counts by time and reporting delay periods. The data can be structured as follows:

| Incident Period | Delay Period 1 | Delay Period 2 | ... | Delay Period N |
|-----------------|----------------|----------------|-----|----------------|
| 1               | 12             | 8              | ... | 3              |
| 2               | 9              | 11             | ... | 2              |
| ...             | ...            | ...            | ... | ...            |

Note: The sample code uses the data dropping the first column ('simulation.csv' does not include the Incident Period column).

## Model Structure

The Bayesian IBNR model is defined using the `nimble` package. Key components of the model include:

- **Negative Binomial Likelihood**: Models incident counts for each time and delay period.
- **Predictors (`g` and `beta` coefficients)**: These model time- and delay-specific effects on reporting probability and incident rate.
- **Hyperparameters (`sig` coefficients)**: Exponentially distributed priors for controlling the variance of the predictors.


## Running the Model

### Running with Parallel Chains

To run the model using multiple parallel chains, the `doParallel` package is used. The model is configured to run with 3 parallel chains. 

The MCMC samples provide posterior distributions for all model parameters, including incident counts and the predictors.
 
## R markdown example

This repository also includes an R Markdown example (bayesianibnr_healthcare.md) demonstrating the application of our model for predicting healthcare data breaches. The model utilizes a negative binomial distribution framework, with parameters estimated using MCMC methods through the nimble package in R. The example file walks users through each step, from loading and preparing synthetic healthcare data to configuring and running the model, interpreting posterior predictions, and visualizing the results.
 
