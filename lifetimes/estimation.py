from __future__ import print_function


import numpy as np
from numpy import log, exp
import pandas as pd

from scipy.special import gammaln, hyp2f1
from scipy.optimize import minimize

from lifetimes.utils import coalesce


class BaseFitter():

    def __repr__(self):
        classname = self.__class__.__name__
        try:
            s = """<lifetimes.%s: fitted with %d customers>""" % (classname, self.data.shape[0])
        except AttributeError:
            s = """<lifetimes.%s>""" % classname
        return s


class BetaGeoFitter(BaseFitter):

    """

    Based on [1], this model has the following assumptions:

    1) Each individual, i, has a hidden lambda_i and p_i parameter
    2) These come from a population wide Gamma and a Beta distribution respectively.
    3) Individuals purchases follow a Poisson process with rate lambda_i*t .
    4) After each purchase, an individual has a p_i probablity of dieing (never buying again).

    [1] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
        "Counting Your Customers the Easy Way: An Alternative to the
        Pareto/NBD Model," Marketing Science, 24 (2), 275-84.

    """

    def fit(self, frequency, recency, cohort, iterative_fitting=1):
        """
        This methods fits the data to the BG/NBD model. 

        Parameters:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            cohort: the cohort vector of customers' purchases (denoted T in literature).
            iterative_fitting: perform `iterative_fitting` additional fits to find the best 
                parameters for the model. Setting to 0 will improve peformance but possibly 
                hurt estimates. 

        Returns:
            self, with additional properties and methods like params_ and plot

        """
        frequency = np.asarray(frequency)
        recency = np.asarray(recency)
        cohort = np.asarray(cohort)
        
        params, self._negative_log_likelihood_ = self._fit(frequency, recency, cohort, iterative_fitting)

        self.params_ = dict(zip(['r', 'alpha', 'a', 'b'], params))
        self.data = pd.DataFrame(np.c_[frequency, recency, cohort], columns=['frequency', 'recency', 'cohort'])
        self.plot = self._plot
        self.plot_period_transactions = self._plot_period_transactions
        return self


    def _fit(self, frequency, recency, cohort, iterative_fitting):
        ll = []
        sols = []
        methods = ['Powell', 'Nelder-Mead']

        for i in range(iterative_fitting + 1):
            fit_method = methods[i%len(methods)]
            params_init = np.random.exponential(1, size=4)
            output = minimize(self._negative_log_likelihood, method=fit_method, tol=1e-8,
                              x0=params_init, args=(frequency, recency, cohort), options={'disp': False})
            ll.append(output.fun)
            sols.append(output.x)

        params = sols[np.argmin(ll)]
        return params, np.min(ll)

    @staticmethod
    def _negative_log_likelihood(params, freq, rec, T):
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
        return -np.sum(A_1 + A_2 + log(exp(A_3) + d * exp(A_4)))

    def _unload_params(self):
        return self.params_['r'], self.params_['alpha'], self.params_['a'], self.params_['b']

    def expected_number_of_purchases_up_to_time(self, t):
        """
        Calculate the expected number of purchases up to time t for a randomly choose individual from
        the population.

        Parameters:
            t: a scalar or array of times.

        Returns: a scalar or array
        """
        r, alpha, a, b = self._unload_params()
        hyp = hyp2f1(r, b, a + b - 1, t / (alpha + t))
        return (a + b - 1) / (a - 1) * (1 - hyp * (alpha / (alpha + t)) ** r) + 1

    def conditional_expected_number_of_purchases_up_to_time(self, t, x, t_x, T):
        """
        Calculate the expected number of purchases up to time t for a randomly choose individual from
        the population, given they have purchase history (x, t_x, T)

        Parameters:
            t: a scalar or array of times.
            x: a scalar: historical frequency of customer.
            t_x: a scalar: historical recency of customer.
            T: a scalar: cohort of the customer.

        Returns: a scalar or array
        """

        r, alpha, a, b = self._unload_params()

        hyp_term = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
        first_term = (a + b + x - 1) / (a - 1)
        second_term = (1 - hyp_term * ((alpha + T) / (alpha + t + T)) ** (r + x))
        numerator = first_term * second_term

        denominator = 1 + (x > 0) * (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x)

        return numerator / denominator

    def _plot(self, **kwargs):
        from matplotlib import pyplot as plt

        ax = plt.subplot(111)
        color_cycle = ax._get_lines.color_cycle

        label = kwargs.pop('label', None)
        color = coalesce(kwargs.pop('c', None), kwargs.pop('color', None), next(color_cycle))
        max_T = self.data['cohort'].max()

        times = np.linspace(0, max_T, 100)
        ax = plt.plot(times, self.expected_number_of_purchases_up_to_time(times), color=color, label=label, **kwargs)

        times = np.linspace(max_T, 1.5*max_T, 100)
        ax = plt.plot(times, self.expected_number_of_purchases_up_to_time(times), color=color, ls='--', **kwargs)

        return ax

    def conditional_probability_alive(self, x, t_x, T):
        """
        Compute the probability that a customer with history (x, t_x, T) is currently
        alive. From http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf

        Parameters:
            x: a scalar: historical frequency of customer.
            t_x: a scalar: historical recency of customer.
            T: a scalar: cohort of the customer.

        Returns: a scalar

        """
        r, alpha, a, b = self._unload_params()

        return 1. / (1 + (x > 0) * (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x))


    def _plot_period_transactions(self, **kwargs):
        from lifetimes.generate_data import beta_geometric_nbd_model
        from matplotlib import pyplot as plt

        bins = kwargs.pop('bins', range(9))
        labels = kwargs.pop('label', ['Actual', 'Model'])

        r, alpha, a, b = self._unload_params()
        n = self.data.shape[0]
        simulated_data = beta_geometric_nbd_model(self.data['cohort'], r, alpha, a, b, size=n)

        ax = plt.hist(np.c_[self.data['frequency'].values, simulated_data['frequency'].values], 
                      bins=bins, label=labels)
        plt.legend()
        plt.xticks(np.arange(len(bins))[:-1]+0.5, bins[:-1])
        plt.title('Frequency of Repeat Transactions')
        plt.ylabel('Customers')
        plt.xlabel('Calibration Period Transactions')
        return ax
