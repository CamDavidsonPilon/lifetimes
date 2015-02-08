
import numpy as np
from scipy import stats
import pandas as pd


def beta_geometric_nbd_model(T, r, alpha, a, b, size=1):
    """
    Generate artificial data according to the BG/NBD model. See [1] for model details


    Parameters:
        T: scalar or array, the length of time observing new customers.
        r, alpha, a, b: scalars, represening parameters in the model. See [1]
        size: the number of customers to generate

    Returns:
        DataFrame, with index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'lambda', 'p', 'alive', 'customer_id'

    [1]: '"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model'
    (http://brucehardie.com/papers/bgnbd_2004-04-20.pdf)

    """

    probability_of_post_purchase_death = stats.beta.rvs(a, b, size=size)
    lambda_ = stats.gamma.rvs(r, scale=1. / alpha, size=size)
    if type(T) in [float, int]:
        T = T * np.ones(size)

    columns = ['frequency', 'recency', 'T', 'lambda', 'p', 'alive', 'customer_id']
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in xrange(size):
        p = probability_of_post_purchase_death[i]
        l = lambda_[i]
        churn = np.argmax(stats.binom.rvs(1, p, size=10 / p)) + 1
        times = np.cumsum(np.r_[[0], stats.expon.rvs(scale=1. / l, size=churn)])
        t_cal = times[times < T[i]]
        df.ix[i] = len(t_cal) - 1,  max(t_cal), T[i], l, p, churn > len(t_cal) - 1, i

    return df.set_index('customer_id')
