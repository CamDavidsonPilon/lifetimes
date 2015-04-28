from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

pd.options.mode.chained_assignment = None

__all__ = ['calibration_and_holdout_data',
           'summary_data_from_transaction_data',
           'calculate_alive_path']


def coalesce(*args):
    return next(s for s in args if s is not None)


def calibration_and_holdout_data(transactions, customer_id_col, datetime_col, calibration_period_end,
                                 observation_period_end=datetime.today(), freq='D', datetime_format=None):
    """
    This function creates a summary of each customer over a calibration and holdout period (training and testing, respectively).
    It accepts transition data, and returns a Dataframe of sufficient statistics.

    Parameters:
        transactions: a Pandas DataFrame of at least two cols.
        customer_id_col: the column in transactions that denotes the customer_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        calibration_period_end: a period to limit the calibration to.
        observation_period_end: a string or datetime to denote the final date of the study. Events
            after this date are truncated.
        datetime_format: a string that represents the timestamp format. Useful if Pandas can't understand
            the provided format.
        freq: Default 'D' for days. Other examples: 'W' for weekly.

    Returns:
        A dataframe with columns frequency_cal, recency_cal, T_cal, frequency_holdout, duration_holdout

    """
    def to_period(d):
        return d.to_period(freq)

    transactions = transactions[[customer_id_col, datetime_col]].copy()

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
    This transforms a Dataframe of transaction data of the form:

        customer_id, datetime

    to a Dataframe of the form:

        customer_id, frequency, recency, T

    Parameters:
        transactions: a Pandas DataFrame.
        customer_id_col: the column in transactions that denotes the customer_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        observation_period_end: a string or datetime to denote the final date of the study. Events
            after this date are truncated.
        datetime_format: a string that represents the timestamp format. Useful if Pandas can't understand
            the provided format.
        freq: Default 'D' for days. Other examples: 'W' for weekly.
    """
    transactions = transactions[[customer_id_col, datetime_col]].copy()

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


def calculate_alive_path(model, transactions, datetime_col, t, freq='D'):
    """
    :param model: A fitted lifetimes model
    :param transactions: a Pandas DataFrame containing the transactions history of the customer_id
    :param datetime_col: the column in the transactions that denotes the datetime the purchase was made
    :param t: the number of time units since the birth for which we want to draw the p_alive
    :param freq: Default 'D' for days. Other examples= 'W' for weekly
    :return: A pandas Series containing the p_alive as a function of T (age of the customer)
    """
    customer_history = transactions[[datetime_col]].copy()
    customer_history[datetime_col] = pd.to_datetime(customer_history[datetime_col])
    customer_history = customer_history.set_index(datetime_col)
    # Add transactions column
    customer_history['transactions'] = 1
    purchase_history = customer_history.resample(freq, how='sum').fillna(0)['transactions'].values
    extra_columns = t - len(purchase_history)
    customer_history = pd.DataFrame(np.append(purchase_history, [0] * extra_columns), columns=['transactions'])
    # add T column
    customer_history['T'] = np.arange(customer_history.shape[0])
    # add cumulative transactions column
    customer_history['frequency'] = customer_history['transactions'].cumsum() - 1  # first purchase is ignored
    # Add t_x column
    customer_history['recency'] = customer_history.apply(lambda row: row['T'] if row['transactions'] != 0 else np.nan, axis=1)
    customer_history['recency'] = customer_history['recency'].fillna(method='ffill').fillna(0)
    return customer_history.apply(lambda row: model.conditional_probability_alive(row['frequency'], row['recency'], row['T']), axis=1)


def _fit(minimizing_function, frequency, recency, T, iterative_fitting, penalizer_coef, initial_params, disp):
    ll = []
    sols = []
    methods = ['Nelder-Mead', 'Powell']

    for i in range(iterative_fitting + 1):
        fit_method = methods[i % len(methods)]
        params_init = np.random.exponential(0.5, size=4) if initial_params is None else initial_params
        output = minimize(minimizing_function, method=fit_method, tol=1e-6,
                          x0=params_init, args=(frequency, recency, T, penalizer_coef), options={'disp': disp})
        ll.append(output.fun)
        sols.append(output.x)
    minimizing_params = sols[np.argmin(ll)]
    return minimizing_params, np.min(ll)


def _scale_time(age):
    # create a scalar such that the maximum age is 10.
    return 10./age.max()


def _check_inputs(frequency, recency, T):

    def check_recency_is_less_than_T(recency, T):
        if np.any(recency > T):
            raise ValueError("""Some values in recency vector are larger than T vector. This is impossible according to the model.""")

    def check_frequency_of_zero_implies_recency_of_zero(frequency, recency):
        ix = frequency == 0
        if np.any(recency[ix] != 0):
            raise ValueError("""There exist non-zero recency values when frequency is zero. This is impossible according to the model.""")

    def check_all_frequency_values_are_integer_values(frequency):
        if np.sum((frequency - frequency.astype(int)) ** 2) != 0:
            raise ValueError("""There exist non-integer values in the frequency vector. This is impossible according to the model.""")

    check_recency_is_less_than_T(recency, T)
    check_frequency_of_zero_implies_recency_of_zero(frequency, recency)
    check_all_frequency_values_are_integer_values(frequency)
