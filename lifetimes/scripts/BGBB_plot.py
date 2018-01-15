import matplotlib.pyplot as plt
import numpy as np
from lifetimes.estimation import BGBBFitter

from lifetimes.plotting import plot_beta
from scipy.stats import beta


def plot_beta_local(g, d):
    plt.plot(np.linspace(0.001, 0.999, 1000), beta.pdf(np.linspace(0.001, 0.999, 1000), g, d))
    plt.title("Beta distribution")
    plt.xlim(0, 1)


def plot_retention(g, d, ts0):
    retention = [1 + BGBBFitter.static_expected_number_of_purchases_up_to_time(a, b, g, d, t) for t in ts0]

    plt.plot(retention)
    plt.title("Retention")

def plot_being_alive(g,d,ts0):
    prob_being_alive = [BGBBFitter.static_probability_of_being_alive(g, d, t) for t in ts0]
    plt.plot(prob_being_alive)
    plt.title("Probability of being alive")
    plt.ylim(0, 1)


a, b = 5, 5  # "gaussian" beta for activity of user

ts0 = range(0, 100)

plt.figure(1)

# first type of beta

g, d = 0.5, 0.5
plt.subplot(331)
plot_beta_local(g, d)

# retention
plt.subplot(332)
plot_retention(g, d, ts0)

# prob of being alive

plt.subplot(333)
plot_being_alive(g,d,ts0)

# second type of beta

g, d = 5, 5
plt.subplot(334)
plot_beta_local(g, d)

# retention
plt.subplot(335)
plot_retention(g, d, ts0)

# prob of being alive

plt.subplot(336)
plot_being_alive(g,d,ts0)

# tjird type of beta

g, d = 0.9, 5
plt.subplot(337)
plot_beta_local(g, d)

# retention
plt.subplot(338)
plot_retention(g, d, ts0)


# prob of being alive

plt.subplot(339)
plot_being_alive(g,d,ts0)
plt.show()
