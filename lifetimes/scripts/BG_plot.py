import numpy as np
import matplotlib.pyplot as plt
from lifetimes.estimation import BGBBFitter, BGFitter

from lifetimes.plotting import plot_beta
from scipy.stats import beta


def plot_beta_local(g, d):
    plt.plot(np.linspace(0.001, 0.999, 1000), beta.pdf(np.linspace(0.001, 0.999, 1000), g, d))
    plt.title("Beta distribution")
    plt.xlim(0, 1)

def plot_renewals(g,d,ts0):
    nb_renewals = [1 + BGFitter.static_expected_number_of_purchases_up_to_time(g, d, t) for t in ts0]
    plt.plot(nb_renewals)
    plt.title("Nb renewals")

def plot_posterior_prob_unsubscribe1(g,d,ts):
    posterior_prob_of_unsubscribe = [np.sum([x * BGFitter.static_unsubscribe_posterior_density(g, d, t, t - 1, x) * 0.01 for x in np.arange(0.01, 1.0, 0.01)])
        for t in ts]

    plt.plot(ts, posterior_prob_of_unsubscribe)
    plt.title("Posterior prob of unsub (t = n + 1) - mean")

def plot_posterior_prob_unsubscribe2(g,d,ts):
    posterior_prob_of_unsubscribe = [np.sum([x * BGFitter.static_unsubscribe_posterior_density(g, d, t, t, x) * 0.01 for x in np.arange(0.01, 1.0, 0.01)])
        for t in ts]

    plt.plot(ts, posterior_prob_of_unsubscribe)
    plt.title("Posterior prob of unsub (t = n) - mean")

limit = 11
ts0 = range(0, limit)
ts = range(1, limit)

plt.figure(1)

# first type of beta

g, d = 0.5, 0.5

plt.subplot(3,4,1)
plot_beta_local(g, d)

# retention
plt.subplot(3,4,2)
plot_renewals(g,d,ts0)

# probability of unsubscribe
plt.subplot(3,4,3)
plot_posterior_prob_unsubscribe1(g,d,ts)

plt.subplot(3,4,4)
plot_posterior_prob_unsubscribe2(g,d,ts)

# second type of beta

g, d = 5, 5
plt.subplot(3,4,5)
plot_beta_local(g, d)

# retention
plt.subplot(3,4,6)
plot_renewals(g,d,ts0)

# probability of unsubscribe
plt.subplot(3,4,7)
plot_posterior_prob_unsubscribe1(g,d,ts)

plt.subplot(3,4,8)
plot_posterior_prob_unsubscribe2(g,d,ts)


# tjird type of beta

g, d = 2, 5

plt.subplot(3,4,9)
plot_beta_local(g, d)

# retention
plt.subplot(3,4,10)
plot_renewals(g,d,ts0)

# probability of unsubscribe
plt.subplot(3,4,11)
plot_posterior_prob_unsubscribe1(g,d,ts)

plt.subplot(3,4,12)
plot_posterior_prob_unsubscribe2(g,d,ts)
plt.show()