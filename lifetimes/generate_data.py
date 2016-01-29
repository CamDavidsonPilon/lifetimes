
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
    if type(T) in [float, int]:
        T = T * np.ones(size)
    else:
        T = np.asarray(T)

    probability_of_post_purchase_death = stats.beta.rvs(a, b, size=size)
    lambda_ = stats.gamma.rvs(r, scale=1. / alpha, size=size)

    columns = ['frequency', 'recency', 'T', 'lambda', 'p', 'alive', 'customer_id']
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        p = probability_of_post_purchase_death[i]
        l = lambda_[i]

        # hacky until I can find something better
        times = []
        next_purchase_in = stats.expon.rvs(scale=1. / l)
        alive = True
        while (np.sum(times) + next_purchase_in < T[i]) and alive:
            times.append(next_purchase_in)
            next_purchase_in = stats.expon.rvs(scale=1. / l)
            alive = np.random.random() > p

        times = np.array(times).cumsum()
        df.ix[i] = len(times), np.max(times if times.shape[0] > 0 else 0), T[i], l, p, alive, i

    return df.set_index('customer_id')


def pareto_nbd_model(T, r, alpha, s, beta, size=1):
    """
    Generate artificial data according to the Pareto/NBD model. See [2] for model details


    Parameters:
        T: scalar or array, the length of time observing new customers.
        r, alpha, s, beta: scalars, representing parameters in the model. See [2]
        size: the number of customers to generate, equal to size of T if T is
           an array.

    Returns:
        DataFrame, with index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'lambda', 'mu', 'alive', 'customer_id'

    [2]: Fader, Peter S. and Bruce G. S. Hardie (2005), "A Note on Deriving the Pareto/NBD Model
    and Related Expressions," <http://brucehardie.com/notes/009/>.

    """
    if type(T) in [float, int]:
        T = T * np.ones(size)
    else:
        T = np.asarray(T)

    lambda_ = stats.gamma.rvs(r, scale=1. / alpha, size=size)
    mus = stats.gamma.rvs(s, scale=1. / beta, size=size)

    columns = ['frequency', 'recency', 'T', 'lambda', 'mu', 'alive', 'customer_id']
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        l = lambda_[i]
        mu = mus[i]
        time_of_death = stats.expon.rvs(scale=1. / mu)

        # hacky until I can find something better
        times = []
        next_purchase_in = stats.expon.rvs(scale=1. / l)
        while np.sum(times) + next_purchase_in < min(time_of_death, T[i]):
            times.append(next_purchase_in)
            next_purchase_in = stats.expon.rvs(scale=1. / l)

        times = np.array(times).cumsum()
        df.ix[i] = len(times), np.max(times if times.shape[0] > 0 else 0), T[i], l, mu, time_of_death > T[i], i

    return df.set_index('customer_id')


def modified_beta_geometric_nbd_model(T, r, alpha, a, b, size=1):
    """
    Generate artificial data according to the MBG/NBD model. See [1,2] for model details
    Parameters:
        T: scalar or array, the length of time observing new customers.
        r, alpha, a, b: scalars, represening parameters in the model. See [1,2]
        size: the number of customers to generate
    Returns:
        DataFrame, with index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'lambda', 'p', 'alive', 'customer_id'
    [1]: '"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model'
    (http://brucehardie.com/papers/bgnbd_2004-04-20.pdf)
    [2] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
        "Empirical validation and comparison of models for customer base analysis,"
        International Journal of Research in Marketing, 24 (3), 201-209.
    """
    if type(T) in [float, int]:
        T = T * np.ones(size)
    else:
        T = np.asarray(T)

    probability_of_post_purchase_death = stats.beta.rvs(a, b, size=size)
    lambda_ = stats.gamma.rvs(r, scale=1. / alpha, size=size)

    columns = ['frequency', 'recency', 'T', 'lambda', 'p', 'alive', 'customer_id']
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        p = probability_of_post_purchase_death[i]
        l = lambda_[i]

        # hacky until I can find something better
        times = []
        next_purchase_in = stats.expon.rvs(scale=1. / l)
        alive = np.random.random() > p  # essentially the difference between this model and BG/NBD
        while (np.sum(times) + next_purchase_in < T[i]) and alive:
            times.append(next_purchase_in)
            next_purchase_in = stats.expon.rvs(scale=1. / l)
            alive = np.random.random() > p

        times = np.array(times).cumsum()
        df.ix[i] = len(times), np.max(times if times.shape[0] > 0 else 0), T[i], l, p, alive, i

    return df.set_index('customer_id')
