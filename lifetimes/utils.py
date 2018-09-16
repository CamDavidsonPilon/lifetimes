"""Lifetimes utils and helpers."""
from __future__ import division

from datetime import datetime

import numpy as np
import pandas as pd
import dill
from scipy.optimize import minimize

pd.options.mode.chained_assignment = None

__all__ = ['calibration_and_holdout_data',
           'summary_data_from_transaction_data',
           '_find_first_transactions',
           'calculate_alive_path',
           'expected_cumulative_transactions']


def calibration_and_holdout_data(transactions, customer_id_col, datetime_col, calibration_period_end,
                                 observation_period_end=None, freq='D', datetime_format=None,
                                 monetary_value_col=None):
    """
    Create a summary of each customer over a calibration and holdout period.

    This function creates a summary of each customer over a calibration and
    holdout period (training and testing, respectively).
    It accepts transaction data, and returns a Dataframe of sufficient statistics.

    Parameters
    ----------
    transactions: :obj: DataFrame
        a Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        the column in transactions DataFrame that denotes the customer_id
    datetime_col:  string
        the column in transactions that denotes the datetime the purchase was made.
    calibration_period_end: :obj: datetime
        a period to limit the calibration to, inclusive.
    observation_period_end: :obj: datetime, optional
         a string or datetime to denote the final date of the study.
         Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    freq: string, optional
        Default 'D' for days. Other examples: 'W' for weekly.
    datetime_format: string, optional
        a string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    monetary_value_col: string, optional
        the column in transactions that denotes the monetary value of the transaction.
        Optional, only needed for customer lifetime value estimation models.

    Returns
    -------
    :obj: DataFrame
        A dataframe with columns frequency_cal, recency_cal, T_cal, frequency_holdout, duration_holdout
        If monetary_value_col isn't None, the dataframe will also have the columns monetary_value_cal and
        monetary_value_holdout.

    """
    def to_period(d):
        return d.to_period(freq)

    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()

    transaction_cols = [customer_id_col, datetime_col]
    if monetary_value_col:
        transaction_cols.append(monetary_value_col)
    transactions = transactions[transaction_cols].copy()

    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format)
    calibration_period_end = pd.to_datetime(calibration_period_end, format=datetime_format)

    # create calibration dataset
    calibration_transactions = transactions.loc[transactions[datetime_col] <= calibration_period_end]
    calibration_summary_data = summary_data_from_transaction_data(calibration_transactions,
                                                                  customer_id_col,
                                                                  datetime_col,
                                                                  datetime_format=datetime_format,
                                                                  observation_period_end=calibration_period_end,
                                                                  freq=freq,
                                                                  monetary_value_col=monetary_value_col)
    calibration_summary_data.columns = [c + '_cal' for c in calibration_summary_data.columns]

    # create holdout dataset
    holdout_transactions = transactions.loc[(observation_period_end >= transactions[datetime_col]) &
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


def _find_first_transactions(transactions, customer_id_col, datetime_col, monetary_value_col=None, datetime_format=None,
                             observation_period_end=None, freq='D'):
    """
    Return dataframe with first transactions.

    This takes a Dataframe of transaction data of the form:
        customer_id, datetime [, monetary_value]
    and appends a column named 'repeated' to the transaction log which indicates which rows
    are repeated transactions for that customer_id.

    Parameters
    ----------
    transactions: :obj: DataFrame
        a Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        the column in transactions DataFrame that denotes the customer_id
    datetime_col:  string
        the column in transactions that denotes the datetime the purchase was made.
    monetary_value_col: string, optional
        the column in transactions that denotes the monetary value of the transaction.
        Optional, only needed for customer lifetime value estimation models.
    observation_period_end: :obj: datetime
        a string or datetime to denote the final date of the study.
        Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    datetime_format: string, optional
        a string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    freq: string, optional
        Default 'D' for days, 'W' for weeks, 'M' for months... etc. Full list here:
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects

    """
    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()

    select_columns = [customer_id_col, datetime_col]

    if monetary_value_col:
        select_columns.append(monetary_value_col)

    transactions = transactions[select_columns].sort_values(select_columns).copy()

    # make sure the date column uses datetime objects, and use Pandas' DateTimeIndex.to_period()
    # to convert the column to a PeriodIndex which is useful for time-wise grouping and truncating
    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    transactions = transactions.set_index(datetime_col).to_period(freq)

    transactions = transactions.loc[(transactions.index <= observation_period_end)].reset_index()

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


def summary_data_from_transaction_data(transactions, customer_id_col, datetime_col, monetary_value_col=None, datetime_format=None,
                                       observation_period_end=None, freq='D', freq_multiplier=1):
    """
    Return summary data from transactions.

    This transforms a Dataframe of transaction data of the form:
        customer_id, datetime [, monetary_value]
    to a Dataframe of the form:
        customer_id, frequency, recency, T [, monetary_value]

    Parameters
    ----------
    transactions: :obj: DataFrame
        a Pandas DataFrame that contains the customer_id col and the datetime col.
    customer_id_col: string
        the column in transactions DataFrame that denotes the customer_id
    datetime_col:  string
        the column in transactions that denotes the datetime the purchase was made.
    monetary_value_col: string, optional
        the columns in the transactions that denotes the monetary value of the transaction.
        Optional, only needed for customer lifetime value estimation models.
    observation_period_end: datetime, optional
         a string or datetime to denote the final date of the study.
         Events after this date are truncated. If not given, defaults to the max 'datetime_col'.
    datetime_format: string, optional
        a string that represents the timestamp format. Useful if Pandas can't understand
        the provided format.
    freq: string, optional
        Default 'D' for days, 'W' for weeks, 'M' for months... etc. Full list here:
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
    freq_multiplier: int, optional
        Default 1, could be use to get exact recency and T, i.e. with freq='W'
        row for user id_sample=1 will be recency=30 and T=39 while data in
        CDNOW summary are different. Exact values could be obtained with
        freq='D' and freq_multiplier=7 which will lead to recency=30.43
        and T=38.86

    Returns
    -------
    :obj: Dataframe:
        customer_id, frequency, recency, T [, monetary_value]

    """
    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()
    observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format).to_period(freq)

    # label all of the repeated transactions
    repeated_transactions = _find_first_transactions(
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

    customers['T'] = (observation_period_end - customers['min']) / freq_multiplier
    customers['recency'] = (customers['max'] - customers['min']) / freq_multiplier

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
    Calculate alive path for plotting alive history of user.

    Parameters
    ----------
    model:
        A fitted lifetimes model
    transactions: :obj: dataframe
        a Pandas DataFrame containing the transactions history of the customer_id
    datetime_col: string
        the column in the transactions that denotes the datetime the purchase was made
    t: array_like
        the number of time units since the birth for which we want to draw the p_alive
    freq: string
        Default 'D' for days. Other examples= 'W' for weekly

    Returns
    -------
    :obj: Series
        A pandas Series containing the p_alive as a function of T (age of the customer)

    """
    customer_history = transactions[[datetime_col]].copy()
    customer_history[datetime_col] = pd.to_datetime(customer_history[datetime_col])
    customer_history = customer_history.set_index(datetime_col)
    # Add transactions column
    customer_history['transactions'] = 1

    # for some reason fillna(0) not working for resample in pandas with python 3.x,
    # changed to replace
    purchase_history = (customer_history.resample(freq).sum().replace(np.nan, 0)
                        ['transactions'].values)

    extra_columns = t + 1 - len(purchase_history)
    customer_history = pd.DataFrame(np.append(purchase_history, [0] * extra_columns), columns=['transactions'])
    # add T column
    customer_history['T'] = np.arange(customer_history.shape[0])
    # add cumulative transactions column
    customer_history['transactions'] = customer_history['transactions'].apply(lambda t: int(t > 0))
    customer_history['frequency'] = customer_history['transactions'].cumsum() - 1  # first purchase is ignored
    # Add t_x column
    customer_history['recency'] = customer_history.apply(lambda row: row['T'] if row['transactions'] != 0 else np.nan, axis=1)
    customer_history['recency'] = customer_history['recency'].fillna(method='ffill').fillna(0)
    return customer_history.apply(
        lambda row: model.conditional_probability_alive(row['frequency'], row['recency'], row['T']),
        axis=1)


def _fit(minimizing_function, minimizing_function_args, iterative_fitting,
         initial_params, params_size, disp, tol=1e-8, fit_method='Nelder-Mead',
         maxiter=2000, **kwargs):
    """Fit function for fitters."""
    ll = []
    sols = []

    def _func_caller(params, func_args, function):
        return function(params, *func_args)

    if iterative_fitting <= 0:
        raise ValueError("iterative_fitting parameter should be greater than 0 as of lifetimes v0.2.1")

    if iterative_fitting > 1 and initial_params is not None:
        raise ValueError("iterative_fitting and initial_params should not be both set, as no improvement could be made.")

    # set options for minimize, if specified in kwargs will be overwrittern
    minimize_options = {}
    minimize_options['disp'] = disp
    minimize_options['maxiter'] = maxiter
    minimize_options.update(kwargs)

    total_count = 0

    while total_count < iterative_fitting:
        current_init_params = np.random.normal(1.0, scale=0.05, size=params_size) if initial_params is None else initial_params
        if minimize_options['disp']:
            print('Optimize function with {}'.format(fit_method))
        output = minimize(_func_caller, method=fit_method, tol=tol,
                          x0=current_init_params,
                          args=(minimizing_function_args, minimizing_function),
                          options=minimize_options)
        sols.append(output.x)
        ll.append(output.fun)

        total_count += 1
    argmin_ll, min_ll = min(enumerate(ll), key=lambda x: x[1])
    minimizing_params = sols[argmin_ll]
    return minimizing_params, min_ll


def _scale_time(age):
    """Create a scalar such that the maximum age is 1."""
    return 1. / age.max()


def _check_inputs(frequency, recency=None, T=None, monetary_value=None):
    """
    Check validity of inputs.

    Raises ValueError when checks failed.

    Parameters
    ----------
    frequency: array_like
        the frequency vector of customers' purchases (denoted x in literature).
    recency: array_like, optional
        the recency vector of customers' purchases (denoted t_x in literature).
    T: array_like, optional
        the vector of customers' age (time since first purchase)
    monetary_value: array_like, optional
        the monetary value vector of customer's purchases (denoted m in literature).

    """
    if recency is not None:
        if T is not None and np.any(recency > T):
            raise ValueError("Some values in recency vector are larger than T vector.")
        if np.any(recency[frequency == 0] != 0):
            raise ValueError("There exist non-zero recency values when frequency is zero.")
        if np.any(recency < 0):
            raise ValueError("There exist negative recency (ex: last order set before first order)")
        if any(len(x) == 0 for x in [recency, frequency, T]):
            raise ValueError("There exists a zero length vector in one of frequency, recency or T.")
    if np.sum((frequency - frequency.astype(int)) ** 2) != 0:
        raise ValueError("There exist non-integer values in the frequency vector.")
    if monetary_value is not None and np.any(monetary_value <= 0):
        raise ValueError("There exist non-positive values in the monetary_value vector.")
    # TODO: raise warning if np.any(freqency > T) as this means that there are
    # more order-periods than periods.


def _customer_lifetime_value(transaction_prediction_model, frequency, recency, T, monetary_value, time=12, discount_rate=0.01):
    """
    Compute the average lifetime value for a group of one or more customers.

    This method computes the average lifetime value for a group of one or more customers.

    Parameters
    ----------
    transaction_prediction_model:
        the model to predict future transactions, literature uses pareto/nbd but we can also use a different model like bg
    frequency: array_like
        the frequency vector of customers' purchases (denoted x in literature).
    recency: array_like
        the recency vector of customers' purchases (denoted t_x in literature).
    T: array_like
        the vector of customers' age (time since first purchase)
    monetary_value: array_like
        the monetary value vector of customer's purchases (denoted m in literature).
    time: int, optional
        the lifetime expected for the user in months. Default: 12
    discount_rate: float, optional
        the monthly adjusted discount rate. Default: 1

    Returns
    -------
    :obj: Series
        series with customer ids as index and the estimated customer lifetime values as values

    """
    df = pd.DataFrame(index=frequency.index)
    df['clv'] = 0  # initialize the clv column to zeros

    for i in range(30, (time * 30) + 1, 30):
        # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
        expected_number_of_transactions = transaction_prediction_model.predict(i, frequency, recency, T) - transaction_prediction_model.predict(i - 30, frequency, recency, T)
        # sum up the CLV estimates of all of the periods
        df['clv'] += (monetary_value * expected_number_of_transactions) / (1 + discount_rate) ** (i / 30)

    return df['clv']  # return as a series


def expected_cumulative_transactions(model, transactions, datetime_col,
                                     customer_id_col, t, datetime_format=None,
                                     freq='D', set_index_date=False,
                                     freq_multiplier=1):
    """
    Get expected and actual repeated cumulative transactions.

    Parameters
    ----------
    model:
        A fitted lifetimes model
    transactions: :obj: DataFrame
        a Pandas DataFrame containing the transactions history of the customer_id
    datetime_col: string
        the column in transactions that denotes the datetime the purchase was made.
    customer_id_col: string
        the column in transactions that denotes the customer_id
    t: int
        the number of time units since the begining of
        data for which we want to calculate cumulative transactions
    datetime_format: string, optional
        a string that represents the timestamp format. Useful if Pandas can't
        understand the provided format.
    freq: string, optional
        Default 'D' for days, 'W' for weeks, 'M' for months... etc. Full list here:
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
    set_index_date: bool, optional
        when True set date as Pandas DataFrame index, default False - number of time units
    freq_multiplier: int, optional
        Default 1, could be use to get exact cumulative transactions predicted
        by model, i.e. model trained with freq='W', passed freq to
        expected_cumulative_transactions is freq='D', and freq_multiplier=7.

    Returns
    -------
    :obj: DataFrame
        A dataframe with columns actual, predicted

    """
    start_date = pd.to_datetime(transactions[datetime_col],
                                format=datetime_format).min()
    start_period = start_date.to_period(freq)
    observation_period_end = start_period + t

    repeated_and_first_transactions = _find_first_transactions(
        transactions,
        customer_id_col,
        datetime_col,
        datetime_format=datetime_format,
        observation_period_end=observation_period_end,
        freq=freq
    )

    first_trans_mask = repeated_and_first_transactions['first']
    repeated_transactions = repeated_and_first_transactions[~first_trans_mask]
    first_transactions = repeated_and_first_transactions[first_trans_mask]

    date_range = pd.date_range(start_date, periods=t + 1, freq=freq)
    date_periods = date_range.to_period(freq)

    pred_cum_transactions = []
    first_trans_size = first_transactions.groupby('date').size()
    for i, period in enumerate(date_periods):
        if i % freq_multiplier == 0 and i > 0:
            times = period - first_trans_size.index
            times = times[times > 0].astype(float) / freq_multiplier
            expected_trans_agg = \
                model.expected_number_of_purchases_up_to_time(times)

            mask = first_trans_size.index < period
            expected_trans = sum(expected_trans_agg * first_trans_size[mask])
            pred_cum_transactions.append(expected_trans)

    act_trans = repeated_transactions.groupby('date').size()
    act_tracking_transactions = act_trans.reindex(date_periods, fill_value=0)

    act_cum_transactions = []
    for j in range(1, t // freq_multiplier + 1):
        sum_trans = sum(act_tracking_transactions.iloc[:j * freq_multiplier])
        act_cum_transactions.append(sum_trans)

    if set_index_date:
        index = date_periods[freq_multiplier - 1: -1:freq_multiplier]
    else:
        index = range(0, t // freq_multiplier)

    df_cum_transactions = pd.DataFrame({'actual': act_cum_transactions,
                                        'predicted': pred_cum_transactions},
                                       index=index)

    return df_cum_transactions


def _save_obj_without_attr(obj, attr_list, path, values_to_save=None):
    """
    Save object with attributes from attr_list.

    Parameters
    ----------
    obj: obj
        Object of class with __dict__ attribute.
    attr_list: list
        List with attributes to exclude from saving to dill object. If empty
        list all attributes will be saved.
    path: str
        Where to save dill object.
    values_to_save: list, optional
        Placeholders for original attributes for saving object. If None will be
        extended to attr_list length like [None] * len(attr_list)

    """
    if values_to_save is None:
        values_to_save = [None] * len(attr_list)

    saved_attr_dict = {}
    for attr, val_save in zip(attr_list, values_to_save):
        if attr in obj.__dict__:
            item = obj.__dict__.pop(attr)
            saved_attr_dict[attr] = item
            setattr(obj, attr, val_save)

    with open(path, 'wb') as out_file:
        dill.dump(obj, out_file)

    for attr, item in saved_attr_dict.items():
        setattr(obj, attr, item)
