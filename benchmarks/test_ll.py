####################################################################
# Imports
####################################################################

import sys

import lifetimes
from lifetimes.datasets import load_transaction_data
from lifetimes.plotting import (
    plot_cumulative_transactions,
    plot_incremental_transactions,
    plot_period_transactions,
    plot_calibration_purchases_vs_holdout_purchases
)

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

####################################################################
# Loading the Data
####################################################################

transaction_data = load_transaction_data()

calibration_period_end = '2014-07-01'
observation_period_end = '2014-12-31'

beginning = pd.to_datetime(transaction_data['date'].min())

summary_cal_holdout = lifetimes.utils.calibration_and_holdout_data(
    transactions           = transaction_data, 
    customer_id_col        = 'id', 
    datetime_col           = 'date',
    calibration_period_end = calibration_period_end,
    observation_period_end = observation_period_end
)

print('Transaction Data Shape:', transaction_data.shape)
print('Cal-Holdout Shape:', '\t', summary_cal_holdout.shape)
print('')

####################################################################
# Fitting the Model
####################################################################

bgf = lifetimes.BetaGeoFitter(
    penalizer_coef = 0.0
)

bgf.fit(
    summary_cal_holdout['frequency_cal'], 
    summary_cal_holdout['recency_cal'], 
    summary_cal_holdout['T_cal']
)

print(bgf.summary)

print(bgf._negative_log_likelihood_)

print(bgf.ll_summary)

print(bgf.solution_iter)

print(bgf.solution_iter_summary)

####################################################################
# Plots
####################################################################

import matplotlib.pyplot as plt

plot_path = 'benchmarks/images/'
img_type = '.svg'

####################################################################
# log_params
####################################################################

plt.plot(bgf.solution_iter)

plt.xlabel('iteration')
plt.ylabel('value of the parameter')
plt.title('Parameters Convergence before Any Backwards Transformations')

plt.legend(bgf.params_names)

plt.savefig(plot_path + 'solution_iter' + img_type)

####################################################################
# r, alpha, a, b
####################################################################

fig = plt.figure(figsize = (15, 15))

nrows, ncols = 2, 2
subplot_counter = 1
for param in bgf.solution_iter_summary.columns:

    plt.subplot(nrows, ncols, subplot_counter)

    plt.plot(bgf.solution_iter_summary[param])

    plt.xlabel('iteration')
    plt.ylabel('value of the parameter')
    plt.title('Iterative Convergence of Parameter {}'.format(param))

    subplot_counter += 1

plt.savefig(plot_path + 'solution_iter_summary' + img_type)


