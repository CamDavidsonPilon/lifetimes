# -*- coding: utf-8 -*-
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
from numpy import random
import pandas as pd


def beta_geometric_nbd_model(T, r, alpha, a, b, size=1):
    """
    Generate artificial data according to the BG/NBD model.

    See [1] for model details

    Parameters
    ----------
    T: array_like
        The length of time observing new customers.
    r, alpha, a, b: float
        Parameters in the model. See [1]_
    size: int, optional
        The number of customers to generate

    Returns
    -------
    DataFrame
        With index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'lambda', 'p', 'alive', 'customer_id'

    References
    ----------
    .. [1]: '"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model'
       (http://brucehardie.com/papers/bgnbd_2004-04-20.pdf)

    """
    if type(T) in [float, int]:
        T = T * np.ones(size)
    else:
        T = np.asarray(T)

    probability_of_post_purchase_death = random.beta(a, b, size=size)
    lambda_ = random.gamma(r, scale=1.0 / alpha, size=size)

    columns = ["frequency", "recency", "T", "lambda", "p", "alive", "customer_id"]
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        p = probability_of_post_purchase_death[i]
        l = lambda_[i]

        # hacky until I can find something better
        times = []
        next_purchase_in = random.exponential(scale=1.0 / l)
        alive = True
        while (np.sum(times) + next_purchase_in < T[i]) and alive:
            times.append(next_purchase_in)
            next_purchase_in = random.exponential(scale=1.0 / l)
            alive = random.random() > p

        times = np.array(times).cumsum()
        df.iloc[i] = (
            np.unique(np.array(times).astype(int)).shape[0],
            np.max(times if times.shape[0] > 0 else 0),
            T[i],
            l,
            p,
            alive,
            i,
        )

    return df.set_index("customer_id")


def beta_geometric_nbd_model_transactional_data(T, r, alpha, a, b, observation_period_end="2019-1-1", freq="D", size=1):
    """
    Generate artificial transactional data according to the BG/NBD model.

    See [1] for model details

    Parameters
    ----------
    T: int, float or array_like
        The length of time observing new customers.
    r, alpha, a, b: float
        Parameters in the model. See [1]_
    observation_period_end: date_like
        The date observation ends
    freq: string, optional
        Default 'D' for days, 'W' for weeks, 'h' for hours
    size: int, optional
        The number of customers to generate

    Returns
    -------
    DataFrame
        The following columns:
        'customer_id', 'date'

    References
    ----------
    .. [1]: '"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model'
       (http://brucehardie.com/papers/bgnbd_2004-04-20.pdf)

    """
    observation_period_end = pd.to_datetime(observation_period_end)

    if type(T) in [float, int]:
        start_date = [observation_period_end - pd.Timedelta(T - 1, unit=freq)] * size
        T = T * np.ones(size)
    else:
        start_date = [observation_period_end - pd.Timedelta(T[i] - 1, unit=freq) for i in range(size)]
        T = np.asarray(T)

    probability_of_post_purchase_death = random.beta(a, b, size=size)
    lambda_ = random.gamma(r, scale=1.0 / alpha, size=size)

    columns = ["customer_id", "date"]
    df = pd.DataFrame(columns=columns)

    for i in range(size):
        s = start_date[i]
        p = probability_of_post_purchase_death[i]
        l = lambda_[i]
        age = T[i]

        purchases = [[i, s - pd.Timedelta(1, unit=freq)]]
        next_purchase_in = random.exponential(scale=1.0 / l)
        alive = True

        while next_purchase_in < age and alive:
            purchases.append([i, s + pd.Timedelta(next_purchase_in, unit=freq)])
            next_purchase_in += random.exponential(scale=1.0 / l)
            alive = random.random() > p

        df = df.append(pd.DataFrame(purchases, columns=columns))

    return df.reset_index(drop=True)


def pareto_nbd_model(T, r, alpha, s, beta, size=1):
    """
    Generate artificial data according to the Pareto/NBD model.

    See [2]_ for model details.

    Parameters
    ----------
    T: array_like
        The length of time observing new customers.
    r, alpha, s, beta: float
        Parameters in the model. See [1]_
    size: int, optional
        The number of customers to generate

    Returns
    -------
    :obj: DataFrame
        with index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'lambda', 'mu', 'alive', 'customer_id'

    References
    ----------
    .. [2]: Fader, Peter S. and Bruce G. S. Hardie (2005), "A Note on Deriving the Pareto/NBD Model
       and Related Expressions," <http://brucehardie.com/notes/009/>.

    """
    if type(T) in [float, int]:
        T = T * np.ones(size)
    else:
        T = np.asarray(T)

    lambda_ = random.gamma(r, scale=1.0 / alpha, size=size)
    mus = random.gamma(s, scale=1.0 / beta, size=size)

    columns = ["frequency", "recency", "T", "lambda", "mu", "alive", "customer_id"]
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        l = lambda_[i]
        mu = mus[i]
        time_of_death = random.exponential(scale=1.0 / mu)

        # hacky until I can find something better
        times = []
        next_purchase_in = random.exponential(scale=1.0 / l)
        while np.sum(times) + next_purchase_in < min(time_of_death, T[i]):
            times.append(next_purchase_in)
            next_purchase_in = random.exponential(scale=1.0 / l)

        times = np.array(times).cumsum()
        df.iloc[i] = (
            np.unique(np.array(times).astype(int)).shape[0],
            np.max(times if times.shape[0] > 0 else 0),
            T[i],
            l,
            mu,
            time_of_death > T[i],
            i,
        )

    return df.set_index("customer_id")


def modified_beta_geometric_nbd_model(T, r, alpha, a, b, size=1):
    """
    Generate artificial data according to the MBG/NBD model.

    See [3]_, [4]_ for model details

    Parameters
    ----------
    T: array_like
        The length of time observing new customers.
    r, alpha, a, b: float
        Parameters in the model. See [1]_
    size: int, optional
        The number of customers to generate

    Returns
    -------
    DataFrame
        with index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'lambda', 'p', 'alive', 'customer_id'

    References
    ----------
    .. [1]: '"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model'
       (http://brucehardie.com/papers/bgnbd_2004-04-20.pdf)
    .. [2] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
       "Empirical validation and comparison of models for customer base analysis,"
       International Journal of Research in Marketing, 24 (3), 201-209.

    """
    if type(T) in [float, int]:
        T = T * np.ones(size)
    else:
        T = np.asarray(T)

    probability_of_post_purchase_death = random.beta(a, b, size=size)
    lambda_ = random.gamma(r, scale=1.0 / alpha, size=size)

    columns = ["frequency", "recency", "T", "lambda", "p", "alive", "customer_id"]
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        p = probability_of_post_purchase_death[i]
        l = lambda_[i]

        # hacky until I can find something better
        times = []
        next_purchase_in = random.exponential(scale=1.0 / l)
        alive = random.random() > p  # essentially the difference between this model and BG/NBD
        while (np.sum(times) + next_purchase_in < T[i]) and alive:
            times.append(next_purchase_in)
            next_purchase_in = random.exponential(scale=1.0 / l)
            alive = random.random() > p

        times = np.array(times).cumsum()
        df.iloc[i] = (
            np.unique(np.array(times).astype(int)).shape[0],
            np.max(times if times.shape[0] > 0 else 0),
            T[i],
            l,
            p,
            alive,
            i,
        )

    return df.set_index("customer_id")


def beta_geometric_beta_binom_model(N, alpha, beta, gamma, delta, size=1):
    """
    Generate artificial data according to the Beta-Geometric/Beta-Binomial
    Model.

    You may wonder why we can have frequency = n_periods, when frequency excludes their
    first order. When a customer purchases something, they are born, _and in the next
    period_ we start asking questions about their alive-ness. So really they customer has
    bought frequency + 1, and been observed for n_periods + 1

    Parameters
    ----------
    N: array_like
        Number of transaction opportunities for new customers.
    alpha, beta, gamma, delta: float
        Parameters in the model. See [1]_
    size: int, optional
        The number of customers to generate

    Returns
    -------
    DataFrame
        with index as customer_ids and the following columns:
        'frequency', 'recency', 'n_periods', 'lambda', 'p', 'alive', 'customer_id'

    References
    ----------
    .. [1] Fader, Peter S., Bruce G.S. Hardie, and Jen Shang (2010),
       "Customer-Base Analysis in a Discrete-Time Noncontractual Setting,"
       Marketing Science, 29 (6), 1086-1108.

    """

    if type(N) in [float, int, np.int64]:
        N = N * np.ones(size)
    else:
        N = np.asarray(N)

    probability_of_post_purchase_death = random.beta(a=alpha, b=beta, size=size)
    thetas = random.beta(a=gamma, b=delta, size=size)

    columns = ["frequency", "recency", "n_periods", "p", "theta", "alive", "customer_id"]
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)
    for i in range(size):
        p = probability_of_post_purchase_death[i]
        theta = thetas[i]

        # hacky until I can find something better
        current_t = 0
        alive = True
        times = []
        while current_t < N[i] and alive:
            alive = random.binomial(1, theta) == 0
            if alive and random.binomial(1, p) == 1:
                times.append(current_t)
            current_t += 1
        # adding in final death opportunity to agree with [1]
        if alive:
            alive = random.binomial(1, theta) == 0
        df.iloc[i] = len(times), times[-1] + 1 if len(times) != 0 else 0, N[i], p, theta, alive, i
    return df
