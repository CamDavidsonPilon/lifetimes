####################################################################
# Imports
####################################################################

import sys

import lifetimes
from lifetimes.datasets import (
    load_transaction_data,
    load_cdnow_summary_data_with_monetary_value
)
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

print('')

####################################################################
# Terminal Variables
####################################################################

# Alias Terminal Options for Testing:
dict_fitters = {
    'beta_geo'       : 'BetaGeoFitter',
    'pareto'         : 'ParetoNBDFitter',
    'modified_beta'  : 'ModifiedBetaGeoFitter',
    'beta_geo_binom' : 'BetaGeoBetaBinomFitter',
    'gamma'          : 'GammaGammaFitter'
}

fitter_type = dict_fitters[sys.argv[1]]

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

summary_with_money_value = load_cdnow_summary_data_with_monetary_value()
summary_with_money_value.head()
returning_customers_summary = summary_with_money_value[summary_with_money_value['frequency'] > 0]

print('Correlation:', returning_customers_summary[['monetary_value', 'frequency']].corr())
print('')

####################################################################
# Fitting the Model
####################################################################

exec(
    '''model = lifetimes.{}(
        penalizer_coef = 0
    )
    '''.format(fitter_type)
)

if fitter_type != 'GammaGammaFitter':

    model.fit(
        summary_cal_holdout['frequency_cal'], 
        summary_cal_holdout['recency_cal'], 
        summary_cal_holdout['T_cal']
    )

else:

    model.fit(
        returning_customers_summary['frequency'],
        returning_customers_summary['monetary_value']
    )

print(model.summary, '\n') # not applicable for the Pareto/NBD fitter

print(model._negative_log_likelihood_, '\n')

print(model.ll_summary, '\n')

print(model.solution_iter, '\n')

print(model.solution_iter_summary, '\n')

####################################################################
# Plots
####################################################################

import matplotlib.pyplot as plt

plot_path = 'benchmarks/images/'
img_type = '.svg'

####################################################################
# log_params
####################################################################

plt.plot(model.solution_iter)

plt.xlabel('iteration')
plt.ylabel('value of the parameter')
plt.title('Parameters Convergence before Any Backwards Transformations')

plt.legend(model.params_names)

plt.savefig(plot_path + 'solution_iter' + img_type)

print('log_params graph done', '\n')

####################################################################
# Transformed Parameters
####################################################################

fig = plt.figure(figsize = (15, 15))

nrows, ncols = 2, 2
subplot_counter = 1
for param in model.solution_iter_summary.columns:

    plt.subplot(nrows, ncols, subplot_counter)

    plt.plot(model.solution_iter_summary[param])

    plt.xlabel('iteration')
    plt.ylabel('value of the parameter')
    plt.title('Iterative Convergence of Parameter {}'.format(param))

    subplot_counter += 1

plt.savefig(plot_path + 'solution_iter_summary' + img_type)

print('transformed params graph done', '\n')