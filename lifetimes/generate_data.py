import numpy as np
from scipy import stats
import pandas as pd


def hello():
    print "hello from MM! :)"


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


def beta_geometric_nbd_model_with_transactions(T, r, alpha, a, b, size=1):
    """
    Generate artificial data according to the BG/NBD model. See [1] for model details


    Parameters:
        T: scalar or array, the length of time observing new customers.
        r, alpha, a, b: scalars, represening parameters in the model. See [1]
        size: the number of customers to generate

    Returns:
        DataFrame, with index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'lambda', 'p', 'alive', 'customer_id'
        and a dictionary containing the time of transactions

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
    transaction_times = {}  # dictionary containing the single transaction times

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
        transaction_times[i] = times

    return df.set_index('customer_id'), transaction_times


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
            alive = np.random.random > p

        times = np.array(times).cumsum()
        df.ix[i] = len(times), np.max(times if times.shape[0] > 0 else 0), T[i], l, p, alive, i

    return df.set_index('customer_id')


def bgbb_model(T, alpha, beta, gamma, delta, size=1, transactional=False):
    """
    Generate artificial data according to the discrete BG/BB model.

    Parameters:
        T: scalar, the length of time observing new customers.
        alpha, beta, gamma, delta: scalars, representing parameters in the model. See [2]
        size: the number of customers to generate, equal to size of T if T is
           an array.

    Returns:
        DataFrame, with index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'p', 'theta', 'alive', 'customer_id'
    """
    if size < 1:
        raise ValueError("size must be positive")

    if alpha <= 0 or beta <= 0 or gamma <= 0 or delta <= 0:
        raise ValueError("Parameters of beta distribution must all be positive")

    if type(T) in [int]:
        T = T * np.ones(size)
    else:
        raise ValueError("Provide a integer T")

    # Generate hidden parameters fo all costumers
    ps = stats.beta.rvs(alpha, beta, size=size)  # probability of purchasing while alive
    thetas = stats.beta.rvs(gamma, delta, size=size)  # probability of dying at the beginning of a time bin
    users = []
    columns = ['frequency', 'recency', 'T', 'p', 'theta', 'alive', 'customer_id']
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        p = ps[i]  # probability of purchasing, if alive
        theta = thetas[i]  # probability of dying

        # initial conditions (buys at 0)
        x = 0
        tx = 0
        alive = True

        # start testing from t = 1
        t = 1
        ts = []
        while t <= T[i]:
            alive = np.random.random() > theta
            if alive:
                purchases = np.random.random() <= p
                if purchases:
                    ts.append(t)
                t += 1
            else:
                break
        if transactional:
            users.append((T[i], ts))
        else:
            if len(ts) > 0:
                tx = max(ts)
            else:
                tx = 0
            df.ix[i] = len(ts), tx, T[i], p, theta, alive, i
    if transactional:
        return users
    else:
        return df.set_index('customer_id')


def bgbb_model_transactional(T, alpha, beta, gamma, delta, size=1):
    """
    Generate artificial data according to the discrete BG/BB model.

    Parameters:
        T: scalar, the length of time observing new customers.
        alpha, beta, gamma, delta: scalars, representing parameters in the model. See [2]
        size: the number of customers to generate, equal to size of T if T is
           an array.

    Returns:
        DataFrame, with index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'p', 'theta', 'alive', 'customer_id'
    """
    if size < 1:
        raise ValueError("size must be positive")

    if alpha <= 0 or beta <= 0 or gamma <= 0 or delta <= 0:
        raise ValueError("Parameters of beta distribution must all be positive")
    # Generate hidden parameters fo all costumers
    ps = stats.beta.rvs(alpha, beta, size=size)  # probability of purchasing while alive
    thetas = stats.beta.rvs(gamma, delta, size=size)  # probability of dying at the beginning of a time bin

    users = []

    for i in range(size):
        p = ps[i]  # probability of purchasing, if alive
        theta = thetas[i]  # probability of dying

        # initial conditions (buys at 0)
        x = 0
        tx = 0

        # start testing from t = 1
        t = 1
        ts = []
        while t <= T:
            alive = np.random.random() > theta
            if alive:
                purchases = np.random.random() <= p
                if purchases:
                    x += 1
                    ts.append(t)
                t += 1
            else:
                break
        users.append((T, ts))

    return users


def bgbbbb_model(T, alpha, beta, gamma, delta, epsilon, zeta, size=1, time_first_purchase=False, death_time=False):
    """
    Generate artificial data according to the discrete BG/BB/BB model (purchases integrated).

    Parameters:
        T: scalar, the length of time observing new customers.
        alpha, beta, gamma, delta, epsilon, zeta: scalars, representing parameters in the model.
        size: the number of customers to generate, equal to size of T if T is
           an array.
        time_first_purchase: if true, adds the time of the first purchase of a user (useful for conversion)

    Returns:
        DataFrame, with index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'frequency_purchases', 'p', 'theta', 'pi', 'alive', 'customer_id'
    """
    if size < 1:
        raise ValueError("size must be positive")

    if alpha <= 0 or beta <= 0 or gamma <= 0 or delta <= 0 or epsilon <= 0 or zeta <= 0:
        raise ValueError("Parameters of beta distribution must all be positive")

    if type(T) in [int]:
        T = T * np.ones(size)
    else:
        raise ValueError("Provide a integer T")

    # Generate hidden parameters fo all costumers
    ps = stats.beta.rvs(alpha, beta, size=size)  # probability of making a session while alive
    thetas = stats.beta.rvs(gamma, delta, size=size)  # probability of dying at the beginning of a time bin
    pis = stats.beta.rvs(epsilon, zeta, size=size)  # probability of purchasing while alive and making a session

    columns = ['frequency', 'recency', 'T', 'frequency_purchases', 'p', 'theta', 'pi', 'alive', 'customer_id']
    if time_first_purchase:
        columns.append('time_first_purchase')
    elif death_time:
        columns.append('death_time')
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        p = ps[i]  # probability of making a session, if alive
        theta = thetas[i]  # probability of dying
        pi = pis[i]  # probability of purchasing, if alive and making a session

        # initial conditions (has a session at 0)
        x = 0
        tx = 0
        alive = True
        xp = 0
        has_purchased = False
        tfp = np.nan  # time_first_purchase
        dt = np.nan  # death_time

        # first possibility of purchasing (the first day)
        purchases = np.random.random() <= pi
        if purchases:
            xp += 1
            if not has_purchased:
                tfp = 0
            has_purchased = True

        # start testing from t = 1
        t = 1
        while t <= T[i]:
            alive = np.random.random() > theta
            if alive:
                has_session = np.random.random() <= p
                if has_session:
                    x += 1
                    tx = t
                    purchases = np.random.random() <= pi  # see whether this monazza buys too
                    if purchases:
                        xp += 1
                        if not has_purchased:
                            tfp = t
                        has_purchased = True
                t += 1
            else:
                dt = t
                break

        if time_first_purchase:
            df.ix[i] = x, tx, T[i], xp, p, theta, pi, alive, i, tfp
        elif death_time:
            df.ix[i] = x, tx, T[i], xp, p, theta, pi, alive, i, dt
        else:
            df.ix[i] = x, tx, T[i], xp, p, theta, pi, alive, i

    return df.set_index('customer_id')


def generate_pareto_data_for_T_N(T, N, params):
    """
    Quick data generator over time
    :param T:       Max T to generate
    :param N:       How many users per T
    :param params:  The pareto params
    :type params:   dict
    :return:        Generated data
    """
    from lifetimes import models
    pareto = models.ParetoNBDModel()
    data = pd.DataFrame()
    for t in range(T + 1):
        new_data = pareto.generateData(t, params, N)
        data = pd.concat([data, new_data])
    return data


def compress_transaction_data(user_actions):
    """
    Takes the results of uncompressed data generation from a model, in the form of dictionary
    [(T,[t1,...tx])]
    and yields compressed lists (ns, Ns) yielding number of observations/successes for each day from install
    ranging from 0 to T_max
    Args:
        user_actions: dictionary [(T,[t1,...tx])]

    Returns: (ns, Ns)

    """

    if len(user_actions) < 1:
        return None, None

    ns = []
    Ns = []
    for action in user_actions:
        T, ts = action

        # increment array size (if needed)
        while len(Ns) <= T:
            ns.append(0)
            Ns.append(0)

        # fill Ns
        Ns = [N + 1 for N in Ns]

        # fill ns
        ns[0] += 1

        for t in ts:
            if t > T:
                raise ValueError("t cannot be larger than T")
            ns[t] += 1

    return ns, Ns
