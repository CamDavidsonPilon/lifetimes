# utils.py
from datetime import datetime
import pandas as pd

def coalesce(*args):
    return next(s for s in args if s is not None)

def summary_data_from_transaction_data(transactions, customer_id_col, datetime_col, format=None,
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

    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=format)
    observation_period_end = pd.to_datetime(observation_period_end, format=format)

    customers = transactions.groupby(customer_id_col)[datetime_col].agg(['max', 'min', 'count'])

    # subtract 1 from count, as we ignore their first order.
    customers['frequency'] = customers['count'] - 1

    def to_floating_freq(x):
        return x.astype(freq_string).astype(float)

    customers['cohort'] = (observation_period_end - customers['min']).map(to_floating_freq)
    customers['recency'] = (observation_period_end - customers['max']).map(to_floating_freq)
    
    # according to Hardie and Fader this is by definition. 
    # http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf
    customers['recency'].ix[customers['frequency'] == 0] = 0 

    return customers[['frequency', 'recency', 'cohort']].astype(float)
