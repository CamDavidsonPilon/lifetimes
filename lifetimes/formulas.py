from scipy import special
import numpy as np


def gamma_ratio(x, a):
    """
    Returns gamma(x+a)/gamma(x)  for x --> inf
    Args:
        x:  point zero
        a:  delta
    """
    if np.isinf(special.gamma(x + a)) or np.isinf(special.gamma(x)):
        return np.sqrt(x / (x + a)) * np.exp(-a) * np.exp((x + a) * np.log(x + a) - x * np.log(x))
    else:
        return special.gamma(x + a) / special.gamma(x)
