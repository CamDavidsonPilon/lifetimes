
import numpy as np
from scipy import stats


def beta_geometric_nbd_model(observational_period, r, alpha, a, b, size=1, random_births=False):
    """
    Generate artificial data according to the BG/NBD model. See [1] for model details


    Parameters:
        observational_period: scalar, the length of time observing new customers.
        r, alpha, a, b: scalars, represening parameters in the model. See [1]
        size: the number of customers to generate
        random_births: If false, all customers are born at time 0. If True, customers can be born
            uniformly throughout the observational period

    Returns:
        frequency: the number of additional puchases made in the period.
        receny: the time since the last purchase
        cohort: the time between the end of the observational period and when we first saw
            the customer.

    [1]: '"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model'
    (http://brucehardie.com/papers/bgnbd_2004-04-20.pdf)

    """

    probability_of_post_purchase_death = stats.beta.rvs(a, b, size=size)
    lambda_ = stats.gamma.rvs(r, scale=1. / alpha, size=size)
    birth_times = np.zeros(size) if not random_births else observational_period * np.random.rand(size)

    def individuals_life(lmbda, p, birth):
        purchases = 0
        current_time = birth
        time_of_next_purchase = current_time + stats.expon.rvs(scale=1. / lmbda)
        alive = True
        while alive and (time_of_next_purchase <= observational_period):
            purchases += 1
            current_time = time_of_next_purchase

            alive = np.random.rand() > p
            time_of_next_purchase += stats.expon.rvs(scale=1. / lmbda)

        return purchases, observational_period - current_time

    frequency = np.zeros(size)
    recency = np.zeros(size)

    for i in range(size):
        frequency[i], recency[i] = individuals_life(lambda_[i], probability_of_post_purchase_death[i], birth_times[i])

    return frequency, recency, observational_period - birth_times
