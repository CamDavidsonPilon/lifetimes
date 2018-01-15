import numpy as np
import matplotlib.pyplot as plt
from lifetimes.plotting import plot_beta


a, b = 11,30 #0.32, 0.85

# plot beta

plot_beta(a, b)
plt.show()

# retention profile