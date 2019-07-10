# -*- coding: utf-8 -*-
"""Lifetimes utils and helpers."""

from __future__ import division
import numpy as np
import pandas as pd
import dill

pd.options.mode.chained_assignment = None

__all__ = [
    "calibration_and_holdout_data",
    "summary_data_from_transaction_data",
    "calculate_alive_path",
    "expected_cumulative_transactions",
    "holdout_data",
    "customer_lifetime_value", 
    "expected_cumulative_clv"
]


class ConvergenceError(ValueError):
    """
    Convergence Error Class.
    """

    pass


def calibration_and_holdout_data(
    transactions,
    customer_id_col,
    datetime_col,
    calibration_period_end,
    observation_period_end=None,
    freq="D",
    datetime_format=None,
    monetary_value_col=None,
    freq_multiplier=1,
    count_intra_period_transaction=False,
    discrete_time = False,
    discrete_recency_T=False
):
    """
    Create a summary of each customer over a calibration and holdout period.

    This function creates a summary of each customer over a calibration and
    holdout period (training and testing, respectively).
    It accepts transaction data, and returns a DataFrame of sufficient statistics.

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
    freq_multiplier: int, optional
        Default 1, could be use to get exact cumulative transactions predicted
        by model, i.e. model trained with freq='W', passed freq to
        expected_cumulative_transactions is freq='D', and freq_multiplier=7.
    count_intra_period_transaction: bool, optional
        Default False.  If True, defines the first transaction as the single first transaction 
        for the customer.  If False, treats the first time period like the other time periods and 
        counts all transactions done in the first period together.
    discrete_time: bool, optional
        Default false.  If true will count recency and n for discrete models.  If false will calculate recency
        and T for continuous time models
    discrete_recency_T: bool, optional
        Default false.  If true will case recency and T to integers.  If false will return calculated
        recency and frequency.
        

    Returns
    -------
    :obj: DataFrame
        A dataframe with columns frequency_cal, recency_cal, T_cal, frequency_holdout, duration_holdout
        If monetary_value_col isn't None, the dataframe will also have the columns monetary_value_cal and
        monetary_value_holdout.
    """

    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()

    transaction_cols = [customer_id_col, datetime_col]
    
    if monetary_value_col:
        transaction_cols.append(monetary_value_col)
    transactions = transactions[transaction_cols].copy()

    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format)
    calibration_period_end = pd.to_datetime(calibration_period_end, format=datetime_format)
    
    calibration_transactions = transactions.loc[transactions[datetime_col] <= calibration_period_end].copy()
    if calibration_transactions.empty:
        raise ValueError(
            "There is no data available. Check the `calibration_period_end` and confirm that values in `transactions` occur prior to those dates."
        )
        
    # create calibration dataset
    calibration_summary_data = summary_data_from_transaction_data(
        calibration_transactions,
        customer_id_col,
        datetime_col,
        datetime_format=datetime_format,
        observation_period_end=calibration_period_end,
        freq=freq,
        monetary_value_col=monetary_value_col,
        freq_multiplier=freq_multiplier,
        count_intra_period_transaction=count_intra_period_transaction,
        discrete_time = discrete_time,
        discrete_recency_T=discrete_recency_T
    )
    
    calibration_summary_data.columns = [c + "_cal" for c in calibration_summary_data.columns]
        
    # create holdout summary dataset
    holdout_summary_data = holdout_data(
        transactions,
        customer_id_col,
        datetime_col,
        calibration_period_end=calibration_period_end,
        observation_period_end=observation_period_end,
        freq=freq,
        datetime_format=datetime_format,
        monetary_value_col=monetary_value_col,
        freq_multiplier=freq_multiplier
    )
    
    #Join calibration and holdout data sets
    combined_data = calibration_summary_data.join(holdout_summary_data, how="left")
    #combined_data = combined_data.reindex(transactions[customer_id_col].unique())
    combined_data.fillna(0, inplace=True)
    
    return combined_data

def _find_first_transactions(
    transactions,
    customer_id_col,
    datetime_col,
    monetary_value_col=None,
    datetime_format=None,
    observation_period_end=None,
    freq="D",
):
    """
    Return dataframe with first transactions.

    This takes a DataFrame of transaction data of the form:
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

    if type(observation_period_end) == pd.Period:
        observation_period_end = observation_period_end.to_timestamp()

    select_columns = [customer_id_col, datetime_col]

    if monetary_value_col:
        select_columns.append(monetary_value_col)

    transactions = transactions[select_columns].sort_values(select_columns).copy()

    # make sure the date column uses datetime objects, and use Pandas' DateTimeIndex.to_period()
    # to convert the column to a PeriodIndex which is useful for time-wise grouping and truncating
    transactions[datetime_col] = (
        pd.Index(
        pd.to_datetime(transactions[datetime_col], format=datetime_format))
                                  .to_period(freq)
                                  .to_timestamp()
    )
    transactions = transactions.loc[(transactions[datetime_col] <= observation_period_end)]
    
    period_groupby = transactions.groupby([customer_id_col, pd.Grouper(freq=freq, key=datetime_col)])
    
    if monetary_value_col:
        period_transactions = period_groupby[monetary_value_col].sum().reset_index()
    else:
        period_transactions = period_groupby.agg(lambda r: 1).reset_index()
        
    # initialize a new column where we will indicate which are the first transactions
    period_transactions["first"] = False
    # find all of the initial transactions and store as an index
    first_transactions = period_transactions.groupby(customer_id_col, sort=True, as_index=False).head(1).index
    # mark the initial transactions as True
    period_transactions.loc[first_transactions, "first"] = True
    select_columns.append("first")
    # reset datetime_col to period
    period_transactions[datetime_col] = pd.Index(period_transactions[datetime_col]).to_period(freq)

    return period_transactions[select_columns]


def summary_data_from_transaction_data(
    transactions,
    customer_id_col,
    datetime_col,
    monetary_value_col=None,
    datetime_format=None,
    observation_period_end=None,
    freq="D",
    freq_multiplier=1,
    count_intra_period_transaction=False,
    discrete_time=False,
    discrete_recency_T=False
):
    """
    Return summary data from transactions.

    This transforms a DataFrame of transaction data of the form:
        customer_id, datetime [, monetary_value]
    to a DataFrame of the form:
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
    count_intra_period_transaction: bool, optional
        Default True.  If true willl calculate frequency including intra period transactions 
        within the first period.  If false, will ignore the events within the first intra period.
        For example if the freq is 'M' and a customer has events on "2019-01-01" and "2019-01-10".
        True would return a frequency of 1 and false would return a frequency of 0.
    discrete_time: bool, optional
        Default false.  If true will count recency and n for discrete models.  If false will calculate recency
        and T for continuous time models
    discrete_recency_T: bool, optional
        Default false.  If true will case recency and T to integers.  If false will return calculated
        recency and frequency.
    Returns
    -------
    :obj: DataFrame:
        customer_id, frequency, recency, T [, monetary_value]
    """

    if observation_period_end is None:
        observation_period_end = (
            pd.to_datetime(transactions[datetime_col].max(), format=datetime_format)
            .to_period(freq)
            .to_timestamp()
        )
    else:
        observation_period_end = (
            pd.to_datetime(observation_period_end, format=datetime_format)
            .to_period(freq)
            .to_timestamp()
        )
        
    transactions = transactions.copy()
    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
        
    # label all of the repeated transactions.  
    repeated_transactions = _find_first_transactions(
        transactions, customer_id_col, datetime_col, monetary_value_col, datetime_format, observation_period_end, freq
    )    
    repeated_transactions[datetime_col] = pd.Index(repeated_transactions[datetime_col]).to_timestamp()
    
    # count all orders by customer.
    customers = (
        repeated_transactions.groupby(customer_id_col, sort=False)[datetime_col]
        .agg(["min", "max", "count"])
    )
    
    #Count all intra-period transactions as individual events
    if count_intra_period_transaction:
        #Count all events in their periods
        customers["frequency"] = (
            transactions.groupby([customer_id_col])
            .agg("count")[datetime_col]
            .rename("frequency") 
            - 1
        )
    else:
        # subtract 1 from count, as we ignore their first transaction.
        customers["frequency"] = customers["count"] - 1
        
    #If continuous time
    if not discrete_time:
        customers["recency"] = (customers["max"] - customers["min"]) / np.timedelta64(1, freq) / freq_multiplier
        customers["T"] = (observation_period_end - customers["min"]) / np.timedelta64(1, freq) / freq_multiplier
        
        #If discrete recency
        if discrete_recency_T:
            customers["recency"] = customers["recency"].round(0)
            customers["T"]  = customers["T"].round(0)
        
        summary_columns = ["frequency", "recency", "T"]
        
        
    #Discrete time calculations
    else:
        customers["recency"] = np.round((customers["max"] - customers["min"].min()) / np.timedelta64(1, freq) / freq_multiplier, 0)
        customers["n"] = np.round((observation_period_end - customers["min"].min()) / np.timedelta64(1, freq) / freq_multiplier, 0)
        
        customers.loc[customers["frequency"]==0, "recency"] = 0
              
        summary_columns = ["frequency", "recency", "n"]
    
                          
    if monetary_value_col:
        #Average all intra-period transactions as individual events
        if count_intra_period_transaction:
            transactions.loc[transactions.groupby([customer_id_col]).head(1).index, monetary_value_col] = np.NaN
            
            customers["monetary_value"] = (
                transactions.groupby(customer_id_col)[monetary_value_col]
                .mean()
                .fillna(0)
            )
        #Average all events in their periods.  All events in first period are NaN
        else:
            repeated_transactions.loc[repeated_transactions['first'], monetary_value_col] = np.NaN
            
            customers["monetary_value"] = (
                repeated_transactions.groupby(customer_id_col)[monetary_value_col]
                .mean()
                .fillna(0)
            )
                
        summary_columns.append("monetary_value")

    return customers[summary_columns].astype(float)


def calculate_alive_path(
    model, 
    transactions, 
    datetime_col, 
    t, 
    freq="D"
):
    """
    Calculate alive path for plotting alive history of user.

    Uses the ``conditional_probability_alive()`` method of the model to achieve the path.

    Parameters
    ----------
    model:
        A fitted lifetimes model
    transactions: DataFrame
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
    customer_history["transactions"] = 1

    # for some reason fillna(0) not working for resample in pandas with python 3.x,
    # changed to replace
    purchase_history = customer_history.resample(freq).sum().replace(np.nan, 0)["transactions"].values

    extra_columns = t + 1 - len(purchase_history)
    customer_history = pd.DataFrame(np.append(purchase_history, [0] * extra_columns), columns=["transactions"])
    # add T column
    customer_history["T"] = np.arange(customer_history.shape[0])
    # add cumulative transactions column
    customer_history["transactions"] = customer_history["transactions"].apply(lambda t: int(t > 0))
    customer_history["frequency"] = customer_history["transactions"].cumsum() - 1  # first purchase is ignored
    # Add t_x column
    customer_history["recency"] = customer_history.apply(
        lambda row: row["T"] if row["transactions"] != 0 else np.nan, axis=1
    )
    customer_history["recency"] = customer_history["recency"].fillna(method="ffill").fillna(0)

    return customer_history.apply(
        lambda row: model.conditional_probability_alive(row["frequency"], row["recency"], row["T"]), axis=1
    )


def _scale_time(
    age
):
    """
    Create a scalar such that the maximum age is 1.
    """

    return 1.0 / age.max()


def _check_inputs(
    frequency, 
    recency=None, 
    T=None, 
    monetary_value=None
):
    """
    Check validity of inputs.

    Raises ValueError when checks failed.

    The checks go sequentially from recency, to frequency and monetary value:

    - recency > T.
    - recency[frequency == 0] != 0)
    - recency < 0
    - zero length vector in frequency, recency or T
    - non-integer values in the frequency vector.
    - non-positive (<= 0) values in the monetary_value vector

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
        if any(x.shape[0] == 0 for x in [recency, frequency, T]):
            raise ValueError("There exists a zero length vector in one of frequency, recency or T.")
    if np.sum((frequency - frequency.astype(int)) ** 2) != 0:
        raise ValueError("There exist non-integer values in the frequency vector.")
    if monetary_value is not None and np.any(monetary_value <= 0):
        raise ValueError("There exist non-positive (<= 0) values in the monetary_value vector.")
    # TODO: raise warning if np.any(freqency > T) as this means that there are
    # more order-periods than periods.


def _customer_lifetime_value(
    transaction_prediction_model, 
    frequency, 
    recency, 
    T, 
    monetary_value, 
    time=12, 
    discount_rate=0.01, 
    freq="D"
):
    """
    Compute the average lifetime value for a group of one or more customers.

    This method computes the average lifetime value for a group of one or more customers.

    It also applies Discounted Cash Flow.

    Parameters
    ----------
    transaction_prediction_model:
        the model to predict future transactions
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
    df["clv"] = 0  # initialize the clv column to zeros

    steps = np.arange(1, time + 1)
    factor = {"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24}[freq]

    for i in steps * factor:
        # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
        expected_number_of_transactions = transaction_prediction_model.predict(
            i, frequency, recency, T
        ) - transaction_prediction_model.predict(i - factor, frequency, recency, T)
        # sum up the CLV estimates of all of the periods and apply discounted cash flow
        df["clv"] += (monetary_value * expected_number_of_transactions) / (1 + discount_rate) ** (i / factor)

    return df["clv"] # return as a series


def expected_cumulative_transactions(
    model,
    transactions,
    datetime_col,
    customer_id_col,
    t,
    datetime_format=None,
    freq="D",
    set_index_date=False,
    freq_multiplier=1
):
    """
    Get expected and actual repeated cumulative transactions.

    Uses the ``expected_number_of_purchases_up_to_time()`` method from the fitted model
    to predict the number of purchases.

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
    
    start_date = pd.to_datetime(transactions[datetime_col], format=datetime_format).min()
    start_period = start_date.to_period(freq)
    observation_period_end = start_period + (t * freq_multiplier)
    
    repeated_and_first_transactions = _find_first_transactions(
            transactions,
            customer_id_col,
            datetime_col,
            datetime_format=datetime_format,
            observation_period_end=observation_period_end,
            freq=freq,
        )

    repeated_and_first_transactions[datetime_col] = pd.Index(repeated_and_first_transactions[datetime_col]).to_timestamp()
    
    repeated_transactions = repeated_and_first_transactions.loc[~repeated_and_first_transactions['first']]
    first_transactions = repeated_and_first_transactions.loc[repeated_and_first_transactions['first']]
    
    #Get dates at proper periodicity
    date_periods = pd.date_range(start_date, periods=t + 1, freq=freq)
    
    #Properly index transactions
    pred_cum_transactions = []
    first_trans_size = first_transactions.groupby(datetime_col).size()
    first_trans_size = first_trans_size.resample(freq).sum()
    
    for i, period in enumerate(date_periods):
        if i % freq_multiplier == 0 and i > 0:
                times = (period - first_trans_size.index) / np.timedelta64(1, freq)
                times = times[times > 0].astype(float) / freq_multiplier
                expected_trans_agg = model.expected_number_of_purchases_up_to_time(times)

                mask = first_trans_size.index < period
                expected_trans = sum(expected_trans_agg * first_trans_size[mask])
                pred_cum_transactions.append(expected_trans)

    valid_dates = pd.Series(repeated_and_first_transactions[datetime_col].unique()).sort_values()
    
    act_trans = repeated_transactions.groupby(datetime_col).size().reindex(valid_dates, fill_value=0)
    act_tracking_transactions = act_trans.resample(freq).sum().reindex(date_periods, fill_value=0)

    act_cum_transactions = []
    for i, period in enumerate(date_periods):
        if i % freq_multiplier == 0 and i > 0:
            sum_trans = act_tracking_transactions.loc[act_tracking_transactions.index < period].sum()
            act_cum_transactions.append(sum_trans)
        
    if set_index_date:
        index = pd.PeriodIndex(date_periods[freq_multiplier - 1 : -1 : freq_multiplier])
    else:
        index = range(0, t // freq_multiplier)

    df_cum_transactions = pd.DataFrame(
        {"actual": act_cum_transactions, "predicted": pred_cum_transactions} , index=index
    )

    return df_cum_transactions

def _save_obj_without_attr(
    obj, 
    attr_list, 
    path, 
    values_to_save=None
):
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

    with open(path, "wb") as out_file:
        dill.dump(obj, out_file)

    for attr, item in saved_attr_dict.items():
        setattr(obj, attr, item)
                                                   
def holdout_data(
    transactions,
    customer_id_col,
    datetime_col,
    calibration_period_end,
    observation_period_end,
    freq="D",
    datetime_format=None,
    monetary_value_col=None,
    freq_multiplier=1
):
    """
    Create a summary of each customer over a calibration and holdout period.

    This function creates a summary of each customer over a calibration and
    holdout period (training and testing, respectively).
    It accepts transaction data, and returns a DataFrame of sufficient statistics.

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
    freq_multiplier: int, optional
        Default 1, could be use to get exact cumulative transactions predicted
        by model, i.e. model trained with freq='W', passed freq to
        expected_cumulative_transactions is freq='D', and freq_multiplier=7.

    Returns
    -------
    :obj: DataFrame
        A dataframe with columns frequency_cal, recency_cal, T_cal, frequency_holdout, duration_holdout
        If monetary_value_col isn't None, the dataframe will also have the columns monetary_value_cal and
        monetary_value_holdout.

    """
    
    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    
    select_columns = [customer_id_col, datetime_col]
    if monetary_value_col:
        select_columns.append(monetary_value_col)
    
    # create holdout dataset
    holdout_transactions = transactions.loc[
        (observation_period_end >= transactions[datetime_col]) & (transactions[datetime_col] > calibration_period_end), select_columns
    ]

    if holdout_transactions.empty:
        raise ValueError(
            "There is no data available. Check the `observation_period_end` and  `calibration_period_end` and confirm that values in `transactions` occur prior to those dates."
        )
    
    #Add dummy row to maintain proper freq
    holdout_transactions = holdout_transactions.append({customer_id_col:np.NaN, 
                                                        datetime_col:transactions[datetime_col].min(), 
                                                        monetary_value_col:np.NaN}, 
                                                       ignore_index=True)
    
    #Get holdout frequency
    holdout_summary_data = (
        holdout_transactions.groupby([customer_id_col, pd.Grouper(freq=freq, key=datetime_col)])
        .agg(lambda r: 1)
        .groupby(level=customer_id_col)
        .count()
    )
    
    holdout_summary_data.columns = ["frequency_holdout"]
    
    #Get holdout monetary average
    if monetary_value_col:
        holdout_summary_data["monetary_value_holdout"] = (
            holdout_transactions.groupby([customer_id_col, pd.Grouper(freq=freq, key=datetime_col)])[monetary_value_col]
            .sum()
            .groupby(level=customer_id_col)
            .mean()
        )
    
    #Reindex by transactions to get those who didn't spend in holdout period
    holdout_summary_data = holdout_summary_data.reindex(transactions[customer_id_col].unique())
    holdout_summary_data.fillna(0, inplace=True)

    #Duration
    observation_period_end = (
            pd.to_datetime(observation_period_end, format=datetime_format)
            .to_period(freq)
            .to_timestamp()
        )
    calibration_period_end = (
            pd.to_datetime(calibration_period_end, format=datetime_format)
            .to_period(freq)
            .to_timestamp()
        )
    
    delta_time = (observation_period_end - calibration_period_end) / np.timedelta64(1, freq) / freq_multiplier
    holdout_summary_data["duration_holdout"] = delta_time
    
    return holdout_summary_data

def expected_cumulative_clv(transaction_prediction_model, 
                            transactions, 
                            datetime_col, 
                            customer_id_col, 
                            monetary_value_col,
                            frequency,
                            recency,
                            T,
                            monetary_value,
                            freq,
                            model_freq,
                            datetime_format=None,
                            set_index_date=True,
                            discount_rate=0,
                            t_start=0,
                            cal_clv=0):
    """
    Get expected and actual repeated cumulative clv.

    Parameters
    ----------
    transaction_prediction_model: lifetimes model
        A fitted lifetimes model for predicting transactions
    spend_prediction_model: lifetimes model
        A fitted lifetimes model for predicting spend
    transactions: :obj: DataFrame
        a Pandas DataFrame containing the transactions history of the customer_id
    datetime_col: string
        the column in transactions that denotes the datetime the purchase was made.
    monetary_value_col: str
        The column in transactions that denotes the monetary_value
    customer_id_col: string
        the column in transactions that denotes the customer_id
    frequency: array_like
        the frequency vector of customers' purchases 
        (denoted x in literature).
    recency: array_like
        the recency vector of customers' purchases
        (denoted t_x in literature).
    T: array_like
        customers' age (time units since first purchase)
    monetary_value: array_like
        the monetary value vector of customer's purchases
        (denoted m in literature).
    freq: str
        Frequency of cumulative clv
    model_freq: str
        Frequency of T for the transaction_prediction_model
    datetime_format: string, optional
        a string that represents the timestamp format. Useful if Pandas can't
        understand the provided format.
    set_index_date: bool, optional
        when True set date as Pandas DataFrame index, default False - number of time units
    freq_multiplier: int, optional
        Default 1, could be use to get exact cumulative transactions predicted
        by model, i.e. model trained with freq='W', passed freq to
        expected_cumulative_transactions is freq='D', and freq_multiplier=7.
    t_start: int, optional
        the time we should start counting CLV
    cal_clv: arrray_like, optional,
        starting clv for customers
    Returns
    -------
    :obj: DataFrame
        A dataframe with columns actual, predicted

    """

    transactions[datetime_col] = (
            pd.Index(
            pd.to_datetime(transactions[datetime_col], format=datetime_format))
                                      .to_period(freq)
                                      .to_timestamp()
        )

    historic_clv = transactions.groupby(datetime_col)[monetary_value_col].sum().cumsum()

    expected_clv = customer_lifetime_value(transaction_prediction_model, 
                                            frequency, 
                                            recency, 
                                            T, 
                                            monetary_value,
                                            time=historic_clv.shape[0],
                                            freq=freq,
                                            model_freq=model_freq,
                                            discount_rate=discount_rate,
                                            t_start=t_start,
                                            cal_clv=cal_clv).sum(axis=0)
    
    if t_start > 0:
        expected_clv = pd.Series(np.append(historic_clv.loc[historic_clv.index <= historic_clv.index[int(t_start)]], expected_clv), index=historic_clv.index)
    else:
        expected_clv.index = historic_clv.index
    

        
    if set_index_date:
        index = historic_clv.index
    else:
        index = range(0, historic_clv.shape[0])

    df_cum_clv = pd.DataFrame({"Holdout":historic_clv,
                               "Predicted":expected_clv},
                 index=index)
    
    return df_cum_clv
    
def customer_lifetime_value(transaction_prediction_model, 
                            frequency, 
                            recency, 
                            T, 
                            monetary_value, 
                            time,
                            freq, 
                            model_freq, 
                            discount_rate=0,
                            t_start=0,
                            cal_clv=0):
    """
    Compute the average lifetime value across time for a group of one or more customers for any freq.

    Parameters
    ----------
    transaction_prediction_model:
        the model to predict future transactions
    frequency: array_like
        the frequency vector of customers' purchases (denoted x in literature).
    recency: array_like
        the recency vector of customers' purchases (denoted t_x in literature).
    T: array_like
        the vector of customers' age (time since first purchase)
    monetary_value: array_like
        the monetary value vector of customer's purchases (denoted m in literature).
    time: int
        the lifetime expected for the user
    freq: str
        Frequency of cumulative clv
    model_freq: str
        Frequency of T for the transaction_prediction_model
    discount_rate: float, optional
        the monthly adjusted discount rate. Default: 0
    t_start: int, optional
        the time we should start counting CLV
    cal_clv: arrray_like, optional,
        starting clv for customers

    Returns
    -------
    :obj: DataFrame
        DataFrame with customer ids as index, steps as columns and the estimated customer lifetime values as values

    """
    df = pd.DataFrame(index=frequency.index)

    factor = {"M":{"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24},
              "W":{"W": 1.0, "M": 1/4.345, "D": 7, "H": 7 * 24},
              "D":{"W": 1/7, "M": 1/30, "D": 1, "H": 1 * 24},
              "D":{"W": 1/(7 * 24), "M": 1/(30 * 24), "D": 24, "H": 1}}[freq][model_freq]

    if t_start > 0:
        steps = np.arange(t_start + 1, time) * factor
    else:
        steps = np.arange(t_start + 1, time + 1) * factor
    
    df[0] = cal_clv 

    for e, i in enumerate(steps):
        expected_number_of_transactions = (
            transaction_prediction_model.predict(i, frequency, recency, T) - 
            transaction_prediction_model.predict(i - factor, frequency, recency, T)
        )
        df[e+1] = ((monetary_value * expected_number_of_transactions) / (1 + discount_rate) ** (i / factor)) + df.iloc[:,e]

    return df.iloc[:,1:]

