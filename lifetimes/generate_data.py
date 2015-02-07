
from scipy import stats


def pareto_nbd_model(r, alpha, s, beta, size=1):
    mu = rgamma(s, beta, size=size)
    lambda_ = rgamma(r,alpha, size=size)
    tau = rexponential(mu)
    x = pm.rpoisson(lambda_*tau)
    return tau, x


def beta_geometric_nbd_model(observational_period, r, alpha, a, b, size=1):

    probability_of_post_purchase_death = stats.beta.rvs(a, b, size=size)
    lambda_ = stats.gamma.rvs(r, alpha, size=size)

    def individuals_life(lmbda, p):
        purchases = 0
        s = stats.expon.rvs(lmbda)
        alive = True
        while (s <= observational_period) and alive:
            purchases +=1
            s += stats.expon.rvs(lmbda)
            alive = np.random.rand() > p

        return purchases, max(observational_period - s,0)

    
