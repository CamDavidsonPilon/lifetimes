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
        return gamma_body(x + a) / gamma_body(x) * np.exp(-a) * np.exp((x + a) * np.log(x + a) - x * np.log(x))
    else:
        return special.gamma(x + a) / special.gamma(x)




def gamma_body(x):
    return np.sqrt(2 * np.pi / x) + 1.0 / 6 * np.sqrt(np.pi / 2) * (1.0 / x) ** (3.0 / 2) + 1.0 / 144 * np.sqrt(
        np.pi / 2) * (1.0 / x) ** (5.0 / 2) - 139.0 / 25920 * np.sqrt(np.pi / 2) * (1.0 / x) ** (
    7.0 / 2) - 571.0 / 1244160 * np.sqrt(np.pi / 2) * (1.0 / x) ** (9.0 / 2)
