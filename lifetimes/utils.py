from datetime import datetime

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

__all__ = ['coalesce', 'calibration_and_holdout_data', 'summary_data_from_transaction_data']


def coalesce(*args):
    return next(s for s in args if s is not None)


def to_floating_freq(x, freq_string):
    return x.astype(freq_string).astype(float)


def calibration_and_holdout_data(transactions, customer_id_col, datetime_col, calibration_period_end, datetime_format=None,
                                 observation_period_end=datetime.today(), freq='D'):
    """
    This function creates a summary of each customer over a calibration and holdout period (training and testing, respectively).
    It accepts transition data, and returns a Dataframe of sufficent statistics.

    Parameters:
        transactions: a Pandas DataFrame of atleast two cols.
        customer_id_col: the column in transactions that denotes the cusomter_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        calibration_period_end: a period to limit the calibration to.

    Returns:
        A dataframe with columns frequency_cal, recency_cal, cohort_cal, frequency_holdout, cohort_holdout

    """

    transactions = transactions.copy()
    freq_string = 'timedelta64[%s]' % freq

    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format)
    calibration_period_end = pd.to_datetime(calibration_period_end, format=datetime_format)

    calibration_transactions = transactions.ix[transactions[datetime_col] <= calibration_period_end]
    holdout_transactions = transactions.ix[transactions[datetime_col] > calibration_period_end]

    calibration_summary_data = summary_data_from_transaction_data(calibration_transactions, customer_id_col, datetime_col,
                                                                  datetime_format, observation_period_end=calibration_period_end, freq=freq)

    holdout_summary_data = summary_data_from_transaction_data(holdout_transactions, customer_id_col, datetime_col,
                                                              datetime_format, observation_period_end=observation_period_end, freq=freq)

    delta_time = to_floating_freq(np.timedelta64(observation_period_end - calibration_period_end), freq_string)
    holdout_summary_data['cohort'] = delta_time
    holdout_summary_data['frequency'] += 1

    combined_data = calibration_summary_data.join(holdout_summary_data, how='left', rsuffix='_holdout', lsuffix='_cal')
    del combined_data['recency_holdout']
    combined_data['frequency_holdout'].fillna(0, inplace=True)
    combined_data['cohort_holdout'] = delta_time

    return combined_data


def summary_data_from_transaction_data(transactions, customer_id_col, datetime_col, datetime_format=None,
                                       observation_period_end=datetime.today(), freq='D'):
    """
    This transform transaction data of the form:
        customer_id, time

        to

        customer_id, frequency, recency, cohort

    Parameters:
        transactions: a Pandas DataFrame of atleast two cols.
        customer_id_col: the column in transactions that denotes the cusomter_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
    """
    transactions = transactions.copy()
    freq_string = 'timedelta64[%s]' % freq

    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format)
    transactions = transactions.ix[transactions[datetime_col] <= observation_period_end]

    customers = transactions.groupby(customer_id_col)[datetime_col].agg(['max', 'min', 'count'])

    # subtract 1 from count, as we ignore their first order.
    customers['frequency'] = customers['count'] - 1

    customers['cohort'] = (observation_period_end - customers['min']).map(lambda r: to_floating_freq(r, freq_string))
    customers['recency'] = (observation_period_end - customers['max']).map(lambda r: to_floating_freq(r, freq_string))

    # according to Hardie and Fader this is by definition.
    # http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf
    customers['recency'].ix[customers['frequency'] == 0] = 0

    return customers[['frequency', 'recency', 'cohort']].astype(float)
