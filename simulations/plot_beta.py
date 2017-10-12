import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta

fig, ax = plt.subplots(1, 1)

a, b = 0.32, 0.85

mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

x = np.linspace(0.001, 0.999, 1000)

ax.plot(x, beta.pdf(x, a, b),
        'r-', lw=5, alpha=0.6, label='beta pdf')

ax.set_xlim(0, 1)

r = beta.rvs(a, b, size=1000)

ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)

print "Enjoy the plot!"

plt.show()
