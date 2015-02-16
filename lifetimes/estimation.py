from __future__ import print_function

import numpy as np
from numpy import log, exp
import pandas as pd

from scipy.special import gammaln, hyp2f1, beta, gamma

from lifetimes.utils import _fit
from lifetimes.generate_data import pareto_nbd_model, beta_geometric_nbd_model

__all__ = ['BetaGeoFitter', 'ParetoNBDFitter']


class BaseFitter():

    def __repr__(self):
        classname = self.__class__.__name__
        try:
            s = """<lifetimes.%s: fitted with %d customers, %s>""" % (classname, self.data.shape[0], self._print_params())
        except AttributeError:
            s = """<lifetimes.%s>""" % classname
        return s

    def _unload_params(self, *args):
        return [self.params_[x] for x in args]

    def _print_params(self):
        s = ""
        for p, value in self.params_.iteritems():
            s += "%s: %.2f, " % (p, value)
        return s.strip(', ')


class ParetoNBDFitter(BaseFitter):

    def __init__(self, penalizer_coef=0.):
        self.penalizer_coef = penalizer_coef

    def fit(self, frequency, recency, T, iterative_fitting=1, initial_params=None):
        """
        This methods fits the data to the Pareto/NBD model.

        Parameters:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            iterative_fitting: perform `iterative_fitting` additional fits to find the best
                parameters for the model. Setting to 0 will improve peformance but possibly
                hurt estimates.

        Returns:
            self, with additional properties and methods like params_ and plot

        """
        frequency = np.asarray(frequency)
        recency = np.asarray(recency)
        T = np.asarray(T)

        params, self._negative_log_likelihood_ = _fit(self._negative_log_likelihood, frequency, recency, T, iterative_fitting, self.penalizer_coef, initial_params)

        self.params_ = dict(zip(['r', 'alpha', 's', 'beta'], params))
        self.data = pd.DataFrame(np.c_[frequency, recency, T], columns=['frequency', 'recency', 'T'])
        self.generate_new_data = lambda size=1: pareto_nbd_model(T, *params, size=size)

        return self

    @staticmethod
    def _A_0(params, freq, rec, T):
        x = freq
        r, alpha, s, beta = params
        max_alpha_beta = max(alpha, beta)
        min_alpha_beta = min(alpha, beta)
        t = s + 1. if alpha > beta else r + x
        r_s_x = r + s + x

        return hyp2f1(r_s_x, t, r_s_x + 1., (max_alpha_beta - min_alpha_beta) / (max_alpha_beta + rec)) / (max_alpha_beta + rec) ** r_s_x\
            - hyp2f1(r_s_x, t, r_s_x + 1., (max_alpha_beta - min_alpha_beta) / (max_alpha_beta + T)) / (max_alpha_beta + T) ** r_s_x

    def _negative_log_likelihood(self, params, freq, rec, T, penalizer_coef):

        if np.any(params <= 0.):
            return np.inf

        r, alpha, s, beta = params
        x = freq

        r_s_x = r + s + x

        A_1 = gammaln(r + x) - gammaln(r) + r * log(alpha) + s * log(beta)
        A_0 = self._A_0(params, freq, rec, T)

        A_2 = log(1. / (alpha + T) ** (r + x) / (beta + T) ** s + (s / r_s_x) * A_0)

        penalizer_term = penalizer_coef * np.log(params).sum()
        return -(A_1 + A_2).sum() + penalizer_term

    def conditional_probability_alive(self, frequency, recency, T):
        """
        Compute the probability that a customer with history (x, t_x, T) is currently
        alive. From http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf

        Parameters:
            frequency: a scalar: historical frequency of customer.
            recency: a scalar: historical recency of customer.
            T: a scalar: age of the customer.

        Returns: a scalar value representing a probability
        """
        x, t_x = frequency, recency
        r, alpha, s, beta = self._unload_params('r', 'alpha', 's', 'beta')

        A_0 = self._A_0(self._unload_params('r', 'alpha', 's', 'beta'), x, t_x, T)
        return 1. / (1. + (s / (r + s + x)) * (alpha + T) ** (r + x) * (beta + T) ** s * A_0)

    def conditional_expected_number_of_purchases_up_to_time(self, t, frequency, recency, T):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population, given they have purchase history (x, t_x, T)

        Parameters:
            t: a scalar or array of times.
            frequency: a scalar: historical frequency of customer.
            recency: a scalar: historical recency of customer.
            T: a scalar: age of the customer.

        Returns: a scalar or array
        """
        x, t_x = frequency, recency
        params = self._unload_params('r', 'alpha', 's', 'beta')
        r, alpha, s, beta = params

        likelihood = exp(-self._negative_log_likelihood(params, x, t_x, T, 0))
        first_term = (gamma(r + x) / gamma(r)) * (alpha ** r * beta ** s) / (alpha + T) ** (r + x) / (beta + T) ** s
        second_term = (r + x) * (beta + T) / (alpha + T) / (s - 1)
        third_term = 1 - ((beta + T) / (beta + T + t)) ** (s - 1)
        return first_term * second_term * third_term / likelihood


class BetaGeoFitter(BaseFitter):

    """

    Also known as the BG/NBD model. Based on [1], this model has the following assumptions:

    1) Each individual, i, has a hidden lambda_i and p_i parameter
    2) These come from a population wide Gamma and a Beta distribution respectively.
    3) Individuals purchases follow a Poisson process with rate lambda_i*t .
    4) After each purchase, an individual has a p_i probability of dieing (never buying again).

    [1] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
        "Counting Your Customers the Easy Way: An Alternative to the
        Pareto/NBD Model," Marketing Science, 24 (2), 275-84.

    """

    def __init__(self, penalizer_coef=0.):
        self.penalizer_coef = penalizer_coef

    def fit(self, frequency, recency, T, iterative_fitting=1, initial_params=None):
        """
        This methods fits the data to the BG/NBD model.

        Parameters:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            iterative_fitting: perform `iterative_fitting` additional fits to find the best
                parameters for the model. Setting to 0 will improve peformance but possibly
                hurt estimates.

        Returns:
            self, with additional properties and methods like params_ and plot

        """
        frequency = np.asarray(frequency)
        recency = np.asarray(recency)
        T = np.asarray(T)

        params, self._negative_log_likelihood_ = _fit(self._negative_log_likelihood, frequency, recency, T, iterative_fitting, self.penalizer_coef, initial_params)

        self.params_ = dict(zip(['r', 'alpha', 'a', 'b'], params))
        self.data = pd.DataFrame(np.c_[frequency, recency, T], columns=['frequency', 'recency', 'T'])
        self.generate_new_data = lambda size=1: beta_geometric_nbd_model(T, *params, size=size)

        return self

    @staticmethod
    def _negative_log_likelihood(params, freq, rec, T, penalizer_coef):
        np.seterr(divide='ignore')

        if np.any(params <= 0):
            return np.inf

        r, alpha, a, b = params

        A_1 = gammaln(r + freq) - gammaln(r) + r * log(alpha)
        A_2 = gammaln(a + b) + gammaln(b + freq) - gammaln(b) - gammaln(a + b + freq)
        A_3 = -(r + freq) * log(alpha + T)

        d = (freq > 0)
        A_4 = log(a) - log(b + freq - 1) - (r + freq) * log(rec + alpha)
        A_4[np.isnan(A_4)] = 0
        penalizer_term = penalizer_coef * np.log(params).sum()
        return -np.sum(A_1 + A_2 + log(exp(A_3) + d * exp(A_4))) + penalizer_term

    def expected_number_of_purchases_up_to_time(self, t):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population.

        Parameters:
            t: a scalar or array of times.

        Returns: a scalar or array
        """
        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')
        hyp = hyp2f1(r, b, a + b - 1, t / (alpha + t))
        return (a + b - 1) / (a - 1) * (1 - hyp * (alpha / (alpha + t)) ** r)

    def conditional_expected_number_of_purchases_up_to_time(self, t, frequency, recency, T):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population, given they have purchase history (x, t_x, T)

        Parameters:
            t: a scalar or array of times.
            frequency: a scalar: historical frequency of customer.
            recency: a scalar: historical recency of customer.
            T: a scalar: age of the customer.

        Returns: a scalar or array
        """
        x, t_x = frequency, recency
        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')

        hyp_term = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
        first_term = (a + b + x - 1) / (a - 1)
        second_term = (1 - hyp_term * ((alpha + T) / (alpha + t + T)) ** (r + x))
        numerator = first_term * second_term

        denominator = 1 + (x > 0) * (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x)

        return numerator / denominator

    def conditional_probability_alive(self, x, t_x, T):
        """
        Compute the probability that a customer with history (x, t_x, T) is currently
        alive. From http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf

        Parameters:
            x: a scalar: historical frequency of customer.
            t_x: a scalar: historical recency of customer.
            T: a scalar: age of the customer.

        Returns: a scalar

        """
        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')

        return 1. / (1 + (x > 0) * (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x))

    def probability_of_purchases_up_to_time(self, t, number_of_purchases):
        """
        Compute the probability of

        P(X(t) = x | model)

        where X(t) is the number of repeat purchases a customer makes in t units of time.
        """

        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')

        x = number_of_purchases
        first_term = beta(a, b + x) / beta(a, b) * gamma(r + x) / gamma(r) / gamma(x + 1) * (alpha / (alpha + t)) ** r * (t / (alpha + t)) ** x
        if x > 0:
            finite_sum = np.sum([gamma(r + j) / gamma(r) / gamma(j + 1) * (t / (alpha + t)) ** j for j in range(0, x)])
            second_term = beta(a + 1, b + x - 1) / beta(a, b) * (1 - (alpha / (alpha + t)) ** r * finite_sum)
        else:
            second_term = 0
        return first_term + second_term
