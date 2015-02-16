from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

pd.options.mode.chained_assignment = None

__all__ = ['calibration_and_holdout_data', 'summary_data_from_transaction_data']


def coalesce(*args):
    return next(s for s in args if s is not None)


def calibration_and_holdout_data(transactions, customer_id_col, datetime_col, calibration_period_end,
                                 observation_period_end=datetime.today(), freq='D', datetime_format=None,):
    """
    This function creates a summary of each customer over a calibration and holdout period (training and testing, respectively).
    It accepts transition data, and returns a Dataframe of sufficent statistics.

    Parameters:
        transactions: a Pandas DataFrame of atleast two cols.
        customer_id_col: the column in transactions that denotes the cusomter_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        calibration_period_end: a period to limit the calibration to.
        observation_period_end: a string or datetime to denote the final date of the study. Events
            after this date are truncated.
        datetime_format: a string that represents the timestamp format. Useful if Pandas can't understand
            the provided format.
        freq: Default 'd' for days. Other examples: 'W' for weekly.

    Returns:
        A dataframe with columns frequency_cal, recency_cal, T_cal, frequency_holdout, duration_holdout

    """
    def to_period(d):
        return d.to_period(freq)

    transactions = transactions.copy()

    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format)
    calibration_period_end = pd.to_datetime(calibration_period_end, format=datetime_format)

    # create calibration dataset
    calibration_transactions = transactions.ix[transactions[datetime_col] <= calibration_period_end]
    calibration_summary_data = summary_data_from_transaction_data(calibration_transactions, customer_id_col, datetime_col,
                                                                  datetime_format, observation_period_end=calibration_period_end, freq=freq)
    calibration_summary_data.columns = [c + '_cal' for c in calibration_summary_data.columns]

    # create holdout dataset
    holdout_transactions = transactions.ix[transactions[datetime_col] > calibration_period_end]
    holdout_transactions[datetime_col] = holdout_transactions[datetime_col].map(to_period)
    holdout_summary_data = reduce_events_to_period(holdout_transactions, customer_id_col, datetime_col).groupby(level=customer_id_col).agg(['count'])
    holdout_summary_data.columns = ['frequency_holdout']

    combined_data = calibration_summary_data.join(holdout_summary_data, how='left')
    combined_data['frequency_holdout'].fillna(0, inplace=True)

    delta_time = to_period(observation_period_end) - to_period(calibration_period_end)
    combined_data['duration_holdout'] = delta_time

    return combined_data


def reduce_events_to_period(transactions, customer_id_col, datetime_col):
    return transactions.groupby([customer_id_col, datetime_col], sort=False).agg(lambda r: 1)


def summary_data_from_transaction_data(transactions, customer_id_col, datetime_col, datetime_format=None,
                                       observation_period_end=datetime.today(), freq='D'):
    """
    This transform transaction data of the form:
        customer_id, time

        to

        customer_id, frequency, recency, T

    Parameters:
        transactions: a Pandas DataFrame of atleast two cols.
        customer_id_col: the column in transactions that denotes the customer_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        observation_period_end: a string or datetime to denote the final date of the study. Events
            after this date are truncated.
        datetime_format: a string that represents the timestamp format. Useful if Pandas can't understand
            the provided format.
        freq: Default 'd' for days. Other examples: 'W' for weekly.
    """
    transactions = transactions.copy()

    def to_period(d):
        return d.to_period(freq)

    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format).map(to_period)
    observation_period_end = to_period(pd.to_datetime(observation_period_end, format=datetime_format))

    transactions = transactions.ix[transactions[datetime_col] <= observation_period_end]

    # reduce all events per customer during the period to a single event:
    period_transactions = reduce_events_to_period(transactions, customer_id_col, datetime_col).reset_index(level=datetime_col)

    # count all orders by customer.
    customers = period_transactions.groupby(level=customer_id_col, sort=False)[datetime_col].agg(['max', 'min', 'count'])

    # subtract 1 from count, as we ignore their first order.
    customers['frequency'] = customers['count'] - 1

    customers['T'] = (observation_period_end - customers['min'])
    customers['recency'] = (customers['max'] - customers['min'])

    return customers[['frequency', 'recency', 'T']].astype(float)


def _fit(minimizing_function, frequency, recency, T, iterative_fitting, penalizer_coef, initial_params):
    ll = []
    sols = []
    methods = ['Powell', 'Nelder-Mead']

    for i in range(iterative_fitting + 1):
        fit_method = methods[i % len(methods)]
        params_init = np.random.exponential(0.5, size=4) if initial_params is None else initial_params
        output = minimize(minimizing_function, method=fit_method, tol=1e-6,
                          x0=params_init, args=(frequency, recency, T, penalizer_coef), options={'disp': False})
        ll.append(output.fun)
        sols.append(output.x)

    minimizing_params = sols[np.argmin(ll)]
    return minimizing_params, np.min(ll)
