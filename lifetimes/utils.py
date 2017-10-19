from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import operator as op
import math

pd.options.mode.chained_assignment = None

__all__ = ['calibration_and_holdout_data',
           'find_first_transactions',
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
        calibration_period_end: a period to limit the calibration to, inclusive.
        observation_period_end: a string or datetime to denote the final date of the study. Events
            after this date are truncated, inclusive.
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
    calibration_summary_data = summary_data_from_transaction_data(calibration_transactions, customer_id_col,
                                                                  datetime_col,
                                                                  datetime_format,
                                                                  observation_period_end=calibration_period_end,
                                                                  freq=freq)
    calibration_summary_data.columns = [c + '_cal' for c in calibration_summary_data.columns]

    # create holdout dataset
    holdout_transactions = transactions.ix[
        (observation_period_end >= transactions[datetime_col]) & (transactions[datetime_col] > calibration_period_end)]
    holdout_transactions[datetime_col] = holdout_transactions[datetime_col].map(to_period)
    holdout_summary_data = reduce_events_to_period(holdout_transactions, customer_id_col, datetime_col).groupby(
        level=customer_id_col).agg(['count'])
    holdout_summary_data.columns = ['frequency_holdout']

    combined_data = calibration_summary_data.join(holdout_summary_data, how='left')
    combined_data['frequency_holdout'].fillna(0, inplace=True)

    delta_time = to_period(observation_period_end) - to_period(calibration_period_end)
    combined_data['duration_holdout'] = delta_time

    return combined_data


def reduce_events_to_period(transactions, *aggregation_columns):
    return transactions.groupby(aggregation_columns, sort=False).agg(lambda r: 1)


def find_first_transactions(transactions, customer_id_col, datetime_col, monetary_value_col=None, datetime_format=None,
                            observation_period_end=datetime.today(), freq='D'):
    """
    This takes a Dataframe of transaction data of the form:
        customer_id, datetime [, monetary_value]
    and appends a column named 'repeated' to the transaction log which indicates which rows
    are repeated transactions for that customer_id.
    Parameters:
        transactions: a Pandas DataFrame.
        customer_id_col: the column in transactions that denotes the customer_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        monetary_value_col: the columns in the transactions that denotes the monetary value of the transaction.
            Optional, only needed for customer lifetime value estimation models.
        observation_period_end: a string or datetime to denote the final date of the study. Events
            after this date are truncated.
        datetime_format: a string that represents the timestamp format. Useful if Pandas can't understand
            the provided format.
        freq: Default 'D' for days, 'W' for weeks, 'M' for months... etc. Full list here:
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
    """
    select_columns = [customer_id_col, datetime_col]

    if monetary_value_col:
        select_columns.append(monetary_value_col)

    transactions = transactions[select_columns].copy()

    # make sure the date column uses datetime objects, and use Pandas' DateTimeIndex.to_period()
    # to convert the column to a PeriodIndex which is useful for time-wise grouping and truncating
    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    transactions = transactions.set_index(datetime_col).to_period(freq)

    transactions = transactions.ix[(transactions.index <= observation_period_end)].reset_index()

    period_groupby = transactions.groupby([datetime_col, customer_id_col], sort=False, as_index=False)

    if monetary_value_col:
        # when we have a monetary column, make sure to sum together any values in the same period
        period_transactions = period_groupby.sum()
    else:
        # by calling head() on the groupby object, the datetime_col and customer_id_col columns
        # will be reduced
        period_transactions = period_groupby.head(1)

    # initialize a new column where we will indicate which are the first transactions
    period_transactions['first'] = False
    # find all of the initial transactions and store as an index
    first_transactions = period_transactions.groupby(customer_id_col, sort=True, as_index=False).head(1).index
    # mark the initial transactions as True
    period_transactions.loc[first_transactions, 'first'] = True
    select_columns.append('first')

    return period_transactions[select_columns]


def summary_data_from_transaction_data(transactions, customer_id_col, datetime_col, monetary_value_col=None,
                                       datetime_format=None,
                                       observation_period_end=datetime.today(), freq='D'):
    """
    This transforms a Dataframe of transaction data of the form:
        customer_id, datetime [, monetary_value]
    to a Dataframe of the form:
        customer_id, frequency, recency, T [, monetary_value]
    Parameters:
        transactions: a Pandas DataFrame.
        customer_id_col: the column in transactions that denotes the customer_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        monetary_value_col: the columns in the transactions that denotes the monetary value of the transaction.
            Optional, only needed for customer lifetime value estimation models.
        observation_period_end: a string or datetime to denote the final date of the study. Events
            after this date are truncated.
        datetime_format: a string that represents the timestamp format. Useful if Pandas can't understand
            the provided format.
        freq: Default 'D' for days, 'W' for weeks, 'M' for months... etc. Full list here:
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
    """
    observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format).to_period(freq)

    # label all of the repeated transactions
    repeated_transactions = find_first_transactions(
        transactions,
        customer_id_col,
        datetime_col,
        monetary_value_col,
        datetime_format,
        observation_period_end,
        freq
    )
    # count all orders by customer.
    customers = repeated_transactions.groupby(customer_id_col, sort=False)[datetime_col].agg(['min', 'max', 'count'])

    # subtract 1 from count, as we ignore their first order.
    customers['frequency'] = customers['count'] - 1

    customers['T'] = (observation_period_end - customers['min'])
    customers['recency'] = (customers['max'] - customers['min'])

    summary_columns = ['frequency', 'recency', 'T']

    if monetary_value_col:
        # create an index of all the first purchases
        first_purchases = repeated_transactions[repeated_transactions['first']].index
        # by setting the monetary_value cells of all the first purchases to NaN,
        # those values will be excluded from the mean value calculation
        repeated_transactions.loc[first_purchases, monetary_value_col] = np.nan

        customers['monetary_value'] = repeated_transactions.groupby(customer_id_col)[monetary_value_col].mean().fillna(
            0)
        summary_columns.append('monetary_value')

    return customers[summary_columns].astype(float)


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
    customer_history['recency'] = customer_history.apply(lambda row: row['T'] if row['transactions'] != 0 else np.nan,
                                                         axis=1)
    customer_history['recency'] = customer_history['recency'].fillna(method='ffill').fillna(0)
    return customer_history.apply(
        lambda row: model.conditional_probability_alive(row['frequency'], row['recency'], row['T']), axis=1)


def _fit(minimizing_function, minimizing_function_args, iterative_fitting, initial_params, params_size, disp,
         jac=False):
    ll = []
    sols = []
    methods = ['Powell', 'Nelder-Mead', 'BFGS']

    def _func_caller(params, func_args, function):
        return function(params, *func_args)

    if jac:
        params_init = np.random.exponential(0.5, size=params_size) if initial_params is None else initial_params
        output = minimize(_func_caller, method='BFGS', tol=1e-6,
                          x0=params_init, args=(minimizing_function_args, minimizing_function), options={'disp': disp},
                          jac=jac)
        ll = output.fun
        minimizing_params = output.x
        return minimizing_params, ll
    else:
        for i in range(iterative_fitting + 1):
            fit_method = methods[i % len(methods)]
            params_init = np.random.exponential(0.5, size=params_size) if initial_params is None else initial_params
            output = minimize(_func_caller, method=fit_method, tol=1e-6,
                              x0=params_init, args=(minimizing_function_args, minimizing_function),
                              options={'disp': disp})
            ll.append(output.fun)
            sols.append(output.x)
        minimizing_params = sols[np.argmin(ll)]
        return minimizing_params, np.min(ll)


def _scale_time(age):
    # create a scalar such that the maximum age is 10.
    return 10. / age.max()


def _check_inputs(frequency, recency, T, N=None, frequency_purchases=None, frequency_before_conversion=None):
    def check_recency_is_less_than_T(recency, T):
        if np.any(recency > T):
            raise ValueError(
                """Some values in recency vector are larger than T vector. This is impossible according to the model.""")

    def check_frequency_of_zero_implies_recency_of_zero(frequency, recency):
        ix = frequency == 0
        if np.any(recency[ix] != 0):
            raise ValueError(
                """There exist non-zero recency values when frequency is zero. This is impossible according to the model.""")

    def check_all_frequency_values_are_integer_values(frequency):
        if np.sum((frequency - frequency.astype(int)) ** 2) != 0:
            raise ValueError(
                """There exist non-integer values in the frequency vector. This is impossible according to the model.""")

    check_recency_is_less_than_T(recency, T)
    check_frequency_of_zero_implies_recency_of_zero(frequency, recency)
    check_all_frequency_values_are_integer_values(frequency)
    if N is not None:
        if np.any(N < 0):
            raise ValueError("""Some values in N are less than zero.""")
    if frequency_purchases is not None:
        check_all_frequency_values_are_integer_values(frequency_purchases)
        if np.any(frequency_purchases > frequency + 1):
            raise ValueError(
                """Some values in frequency_purchases vector are larger than frequency + 1 vector. This is impossible according to the model.""")

    if frequency_before_conversion is not None:
        check_all_frequency_values_are_integer_values(frequency_before_conversion)
        if np.any(frequency_before_conversion > frequency + 1):
            raise ValueError(
                """Some values in frequency_before_conversion vector are larger than frequency vector. This is impossible according to the model.""")

def customer_lifetime_value(transaction_prediction_model, frequency, recency, T, monetary_value, time=12,
                            discount_rate=1):
    """
    This method computes the average lifetime value for a group of one or more customers.
        transaction_prediction_model: the model to predict future transactions, literature uses
            pareto/ndb but we can also use a different model like bg
        frequency: the frequency vector of customers' purchases (denoted x in literature).
        recency: the recency vector of customers' purchases (denoted t_x in literature).
        T: the vector of customers' age (time since first purchase)
        monetary_value: the monetary value vector of customer's purchases (denoted m in literature).
        time: the lifetime expected for the user in months. Default: 12
        discount_rate: the monthly adjusted discount rate. Default: 1

    Returns:
        Series object with customer ids as index and the estimated customer lifetime values as values
    """
    df = pd.DataFrame(index=frequency.index)
    df['clv'] = 0  # initialize the clv column to zeros

    for i in range(30, (time * 30) + 1, 30):
        # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
        expected_number_of_transactions = transaction_prediction_model.predict(i, frequency, recency,
                                                                               T) - transaction_prediction_model.predict(
            i - 30, frequency, recency, T)
        # sum up the CLV estimates of all of the periods
        df['clv'] += (monetary_value * expected_number_of_transactions) / (1 + discount_rate) ** (i / 30)

    return df['clv']  # return as a series


def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom


def normalize_positive_vector(v):
    """
    Normalizes a vector with positive coordinates to the sum of coordinates
    """
    for c in v:
        if c < 0:
            raise ValueError("Negative coordinate!")

    norm = sum(v)
    return [float(c) / norm for c in v]


def is_almost_equal(v1, v2, tol = 0.0001):
    return math.fabs(v1 - v2) < tol


def is_same_order(v1, v2, tol=10):
    if math.fabs(v2) > 0.0001:
        return 1.0/tol < math.fabs(float(v1)/v2) < tol
    elif math.fabs(v1) > 0.0001:
        return 1.0/tol < math.fabs(float(v2)/v1) < tol
    else:
        return True


def multinomial_sample(data):
    """

    Args:
        data: an array t containing a number of users
    """
    total = sum(data)
    prob = [float(n) / total for n in data]
    return np.random.multinomial(total, prob, size=1)[0]

