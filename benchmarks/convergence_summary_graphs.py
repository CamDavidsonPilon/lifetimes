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

print('Transaction Data Shape: {}'.format(transaction_data.shape))
print('Cal-Holdout Shape: \t {}'.format(summary_cal_holdout.shape))
print('')

summary_with_money_value = load_cdnow_summary_data_with_monetary_value()
summary_with_money_value.head()
returning_customers_summary = summary_with_money_value[summary_with_money_value['frequency'] > 0]

print('Correlation: {}'.format(returning_customers_summary[['monetary_value', 'frequency']].corr()))
print('')

####################################################################
# Fitting the Model
####################################################################

# The beta_geo_binom needs a bit of a penalty to converge.
penalizer_coef = 0 if fitter_type != 'BetaGeoBetaBinomFitter' else 0.2
exec(
    '''model = lifetimes.{}(
        penalizer_coef = {}
    )
    '''.format(fitter_type, penalizer_coef)
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

if fitter_type != 'ParetoNBDFitter':
    print(model.summary, '\n') # not applicable for the Pareto/NBD fitter

print('{}\n'.format(model._negative_log_likelihood_))

print('{}\n'.format(model.ll_summary))

print('{}\n'.format(model.solution_iter))

print('{}\n'.format(model.solution_iter_summary))

####################################################################
# Plots
####################################################################

import matplotlib.pyplot as plt

plot_path = 'benchmarks/images/'
img_type = '.svg'

####################################################################
# log_params
####################################################################

def plot_fitter_log_params(
    model,
    xlabel="iteration",
    ylabel="value of the parameter",
    title="Parameters Convergence before Any Transformations",
    ax=None,
    figsize=(8, 6),
    **kwargs
):
    """
    Plots the fitter's approximated log of the parameters convergence.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model, for now only for BG/NBD
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    ax: matplotlib.AxesSubplot, optional
        Using user axes
    figsize: tuple
        size of the image
    kwargs
        Passed into the pandas.DataFrame.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot
    """

    from matplotlib import pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = figsize)

    plt.plot(model.solution_iter)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.legend(model.params_names)

    return ax

ax = plot_fitter_log_params(model = model)

plt.savefig(plot_path + 'solution_iter' + img_type)

print('log_params graph done\n')

####################################################################
# Transformed Parameters
####################################################################

def plot_fitter_params(
    model,
    xlabel="iteration",
    ylabel="value of the parameter",
    title="Iterative Convergence of the Fitter's Parameters",
    figsize=(15, 15),
    ax=None,
    **kwargs
):
    """
    Plots the fitter's approximated parameters convergence.

    Parameters
    ----------
    model: lifetimes model
        A fitted lifetimes model, for now only for BG/NBD
    title: str, optional
        Figure title
    xlabel: str, optional
        Figure xlabel
    ylabel: str, optional
        Figure ylabel
    ax: matplotlib.AxesSubplot, optional
        Using user axes
    figsize: tuple
        size of the image
    kwargs
        Passed into the pandas.DataFrame.plot command.

    Returns
    -------
    axes: matplotlib.AxesSubplot
    """

    from matplotlib import pyplot as plt

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize = figsize)

    subplot_counter = 0
    params = model.solution_iter_summary.columns
    for i in range(nrows):
        for j in range(ncols):

            if subplot_counter < len(params):
                ax = axes[i, j]
                param = params[subplot_counter]

                ax.plot(
                    model.solution_iter_summary[param],
                    label = param,
                )

                ax.set_xlabel('iteration')
                ax.set_ylabel('value of the parameter')
                ax.set_title('Iterative Convergence of Parameter {}'.format(param))

                subplot_counter += 1

    return axes

axes = plot_fitter_params(model = model)

plt.savefig(plot_path + 'solution_iter_summary' + img_type)

print('transformed params graph done\n')

####################################################################
# Graphs with the plotting.py file
####################################################################

# Simply copying the above plotting functions into plotting.py

from lifetimes.plotting import (
    plot_fitter_log_params,
    plot_fitter_params
)

ax = plot_fitter_log_params(model = model)

plt.savefig(plot_path + 'solution_iter_lifetimes' + img_type)

print('log_params graph done\n')

axes = plot_fitter_params(model = model)

plt.savefig(plot_path + 'solution_iter_summary_lifetimes' + img_type)

print('transformed params graph done\n')