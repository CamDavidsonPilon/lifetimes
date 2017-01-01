from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import fmin

pd.options.mode.chained_assignment = None

__all__ = ['calibration_and_holdout_data',
           'find_first_transactions',
           'summary_data_from_transaction_data',
           'calculate_alive_path']


def coalesce(*args):
    return next(s for s in args if s is not None)


def calibration_and_holdout_data(transactions, customer_id_col, datetime_col, calibration_period_end,
                                 observation_period_end=datetime.today(), freq='D', datetime_format=None,
                                 monetary_value_col=None):
    """
    This function creates a summary of each customer over a calibration and holdout period (training and testing,
    respectively).
    It accepts transition data, and returns a Dataframe of sufficient statistics.

    Parameters:
        transactions: a Pandas DataFrame of at least two cols.
        customer_id_col: the column in transactions that denotes the customer_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        calibration_period_end: a period to limit the calibration to, inclusive.
        observation_period_end: a string or datetime to denote the final date of the study. Events
            after this date are truncated, inclusive.
        freq: Default 'D' for days. Other examples: 'W' for weekly.
        datetime_format: a string that represents the timestamp format. Useful if Pandas can't understand
            the provided format.
        monetary_value_col: the column in transactions that denotes the monetary value of the transaction.
            Optional, only needed for customer lifetime value estimation models.

    Returns:
        A dataframe with columns frequency_cal, recency_cal, T_cal, frequency_holdout, duration_holdout
        If monetary_value_col isn't None, the dataframe will also have the columns monetary_value_cal and
        monetary_value_holdout.
    """
    def to_period(d):
        return d.to_period(freq)

    transaction_cols = [customer_id_col, datetime_col]
    if monetary_value_col:
        transaction_cols.append(monetary_value_col)
    transactions = transactions[transaction_cols].copy()

    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format)
    calibration_period_end = pd.to_datetime(calibration_period_end, format=datetime_format)

    # create calibration dataset
    calibration_transactions = transactions.ix[transactions[datetime_col] <= calibration_period_end]
    calibration_summary_data = summary_data_from_transaction_data(calibration_transactions,
                                                                  customer_id_col,
                                                                  datetime_col,
                                                                  datetime_format=datetime_format,
                                                                  observation_period_end=calibration_period_end,
                                                                  freq=freq,
                                                                  monetary_value_col=monetary_value_col)
    calibration_summary_data.columns = [c + '_cal' for c in calibration_summary_data.columns]

    # create holdout dataset
    holdout_transactions = transactions.ix[(observation_period_end >= transactions[datetime_col]) &
                                           (transactions[datetime_col] > calibration_period_end)]
    holdout_transactions[datetime_col] = holdout_transactions[datetime_col].map(to_period)
    holdout_summary_data = holdout_transactions.groupby([customer_id_col, datetime_col], sort=False).agg(lambda r: 1)\
                                               .groupby(level=customer_id_col).agg(['count'])
    holdout_summary_data.columns = ['frequency_holdout']
    if monetary_value_col:
        holdout_summary_data['monetary_value_holdout'] = \
            holdout_transactions.groupby(customer_id_col)[monetary_value_col].mean()

    combined_data = calibration_summary_data.join(holdout_summary_data, how='left')
    combined_data.fillna(0, inplace=True)

    delta_time = to_period(observation_period_end) - to_period(calibration_period_end)
    combined_data['duration_holdout'] = delta_time

    return combined_data


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
        monetary_value_col: the column in transactions that denotes the monetary value of the transaction.
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
    transactions['orders'] = 1

    # make sure the date column uses datetime objects, and use Pandas' DateTimeIndex.to_period()
    # to convert the column to a PeriodIndex which is useful for time-wise grouping and truncating
    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    transactions = transactions.set_index(datetime_col).to_period(freq)

    transactions = transactions.ix[(transactions.index <= observation_period_end)].reset_index()

    period_groupby = transactions.groupby([customer_id_col, datetime_col], sort=False, as_index=False)
    period_transactions = period_groupby.sum()

    # initialize a new column where we will indicate which are the first transactions
    period_transactions['first'] = False
    # find all of the initial transactions and store as an index
    first_transactions = period_transactions.groupby(customer_id_col, sort=True, as_index=False).head(1).index
    # mark the initial transactions as True
    period_transactions.loc[first_transactions, 'first'] = True
    select_columns.extend(['first', 'orders'])

    return period_transactions[select_columns]


def summary_data_from_transaction_data(transactions, customer_id_col, datetime_col, monetary_value_col=None, datetime_format=None,
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

    # sum all orders by customer.
    customers = repeated_transactions.groupby(customer_id_col, sort=False).agg({'orders': ['sum'], datetime_col: ['min', 'max']})
    customers.columns = customers.columns.get_level_values(1)

    # subtract 1 from count, as we ignore their first order.
    customers['frequency'] = customers['sum'] - 1

    customers['T'] = (observation_period_end - customers['min'])
    customers['recency'] = (customers['max'] - customers['min'])

    summary_columns = ['frequency', 'recency', 'T']

    if monetary_value_col:
        # create an index of all the first purchases
        first_purchases = repeated_transactions[repeated_transactions['first']].index
        # by setting the monetary_value cells of all the first purchases to NaN,
        # those values will be excluded from the mean value calculation
        repeated_transactions.loc[first_purchases, monetary_value_col] = np.nan

        customers['monetary_value'] = repeated_transactions.groupby(customer_id_col)[monetary_value_col].mean().fillna(0)
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
    customer_history['recency'] = customer_history.apply(lambda row: row['T'] if row['transactions'] != 0 else np.nan, axis=1)
    customer_history['recency'] = customer_history['recency'].fillna(method='ffill').fillna(0)
    return customer_history.apply(lambda row: model.conditional_probability_alive(row['frequency'], row['recency'], row['T']), axis=1)


def _fit(minimizing_function, minimizing_function_args, iterative_fitting, initial_params, params_size, disp, tol=1e-8):
    ll = []
    sols = []

    def _func_caller(params, func_args, function):
        return function(params, *func_args)

    if iterative_fitting <= 0:
        raise ValueError("iterative_fitting parameter should be greater than 0 as of lifetimes v0.2.1")

    current_init_params = np.random.normal(1.0, scale=0.05, size=params_size) if initial_params is None else initial_params
    total_count = 0
    while total_count < iterative_fitting:
        xopt, fopt, _, _, _ = fmin(_func_caller, current_init_params, ftol=tol,
                                   args=(minimizing_function_args, minimizing_function), disp=disp, maxiter=2000, maxfun=2000, full_output=True)
        sols.append(xopt)
        ll.append(fopt)

        total_count += 1
    argmin_ll, min_ll = min(enumerate(ll), key=lambda x: x[1])
    minimizing_params = sols[argmin_ll]
    return minimizing_params, min_ll


def _scale_time(age):
    # create a scalar such that the maximum age is 10.
    return 10. / age.max()


def _check_inputs(frequency, recency=None, T=None, monetary_value=None):
    if recency is not None:
        if T is not None and np.any(recency > T):
            raise ValueError("Some values in recency vector are larger than T vector.")
        if np.any(recency[frequency == 0] != 0):
            raise ValueError("There exist non-zero recency values when frequency is zero.")
        if np.any(recency < 0):
            raise ValueError("There exist negative recency (ex: last order set before first order)")
    if np.sum((frequency - frequency.astype(int)) ** 2) != 0:
        raise ValueError("There exist non-integer values in the frequency vector.")
    if monetary_value is not None and np.any(monetary_value <= 0):
        raise ValueError("There exist non-positive values in the monetary_value vector.")
    # TODO: raise warning if np.any(freqency > T) as this means that there are
    # more order-periods than periods.


def customer_lifetime_value(transaction_prediction_model, frequency, recency, T, monetary_value, time=12, discount_rate=0.01):
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
        expected_number_of_transactions = transaction_prediction_model.predict(i, frequency, recency, T) - transaction_prediction_model.predict(i - 30, frequency, recency, T)
        # sum up the CLV estimates of all of the periods
        df['clv'] += (monetary_value * expected_number_of_transactions) / (1 + discount_rate) ** (i / 30)

    return df['clv']  # return as a series
