from __future__ import print_function


import numpy as np
from numpy import log, exp
import pandas as pd

from scipy.special import gammaln, hyp2f1, beta, gamma

from lifetimes.utils import coalesce, _fit

__all__ = ['BetaGeoFitter']


class BaseFitter():

    def __repr__(self):
        classname = self.__class__.__name__
        try:
            s = """<lifetimes.%s: fitted with %d customers>""" % (classname, self.data.shape[0])
        except AttributeError:
            s = """<lifetimes.%s>""" % classname
        return s

    def _unload_params(self, *args):
        return [self.params_[x] for x in args]


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

        params, self._negative_log_likelihood_ = _fit(self._negative_log_likelihood, frequency, recency, cohort, iterative_fitting, self.penalizer_coef)

        self.params_ = dict(zip(['r', 'alpha', 'a', 'b'], params))
        self.data = pd.DataFrame(np.c_[frequency, recency, cohort], columns=['frequency', 'recency', 'cohort'])

        # stick on the plotting methods
        self.plot_expected_repeat_purchases = self._plot_expected_repeat_purchases
        self.plot_period_transactions = self._plot_period_transactions
        self.plot_calibration_purchases_vs_holdout_purchases = self._plot_calibration_purchases_vs_holdout_purchases
        self.plot_frequency_recency_matrix = self._plot_frequency_recency_matrix
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

    def conditional_expected_number_of_purchases_up_to_time(self, t, x, t_x, T):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population, given they have purchase history (x, t_x, T)

        Parameters:
            t: a scalar or array of times.
            x: a scalar: historical frequency of customer.
            t_x: a scalar: historical recency of customer.
            T: a scalar: cohort of the customer.

        Returns: a scalar or array
        """

        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')

        hyp_term = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
        first_term = (a + b + x - 1) / (a - 1)
        second_term = (1 - hyp_term * ((alpha + T) / (alpha + t + T)) ** (r + x))
        numerator = first_term * second_term

        denominator = 1 + (x > 0) * (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x)

        return numerator / denominator

    def _plot_expected_repeat_purchases(self, **kwargs):
        from matplotlib import pyplot as plt

        ax = kwargs.pop('ax', None) or plt.subplot(111)
        color_cycle = ax._get_lines.color_cycle

        label = kwargs.pop('label', None)
        color = coalesce(kwargs.pop('c', None), kwargs.pop('color', None), next(color_cycle))
        max_T = self.data['cohort'].max()

        times = np.linspace(0, max_T, 100)
        ax = plt.plot(times, self.expected_number_of_purchases_up_to_time(times), color=color, label=label, **kwargs)

        times = np.linspace(max_T, 1.5 * max_T, 100)
        plt.plot(times, self.expected_number_of_purchases_up_to_time(times), color=color, ls='--', **kwargs)

        plt.title('Expected Number of Repeat Purchases per Customer')
        plt.xlabel('Time Since First Purchase')
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
        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')

        return 1. / (1 + (x > 0) * (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x))

    def _plot_period_transactions(self, **kwargs):
        from lifetimes.generate_data import beta_geometric_nbd_model
        from matplotlib import pyplot as plt

        bins = kwargs.pop('bins', range(9))
        labels = kwargs.pop('label', ['Actual', 'Model'])

        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')
        n = self.data.shape[0]
        simulated_data = beta_geometric_nbd_model(self.data['cohort'], r, alpha, a, b, size=n)

        ax = plt.hist(np.c_[self.data['frequency'].values, simulated_data['frequency'].values],
                      bins=bins, label=labels)
        plt.legend()
        plt.xticks(np.arange(len(bins))[:-1] + 0.5, bins[:-1])
        plt.title('Frequency of Repeat Transactions')
        plt.ylabel('Customers')
        plt.xlabel('Number of Calibration Period Transactions')
        return ax

    def _plot_calibration_purchases_vs_holdout_purchases(self, calibration_holdout_matrix, n=7):
        """
        This currently relies too much on the lifetimes.util calibration_and_holdout_data function.

        """

        from matplotlib import pyplot as plt

        summary = calibration_holdout_matrix.copy()
        T = summary.iloc[0]['cohort_holdout']

        summary['model'] = summary.apply(lambda r: self.conditional_expected_number_of_purchases_up_to_time(T, r['frequency_cal'], r['recency_cal'], r['cohort_cal']), axis=1)

        ax = summary.groupby('frequency_cal')[['frequency_holdout', 'model']].mean().ix[:n].plot()

        plt.title('Actual Purchases in Holdout Period vs Predicted Purchases')
        plt.xlabel('Puchases in Calibration Period')
        plt.ylabel('Average of Purchases in Holdout Period')
        plt.legend()

        return ax

    def _plot_frequency_recency_matrix(self, max_x=None, max_t=None, **kwargs):
        from matplotlib import pyplot as plt
        from .plotting import forceAspect

        if max_x is None:
            max_x = int(self.data['frequency'].max())

        if max_t is None:
            max_t = int(self.data['cohort'].max())

        t = 1  # one unit of time
        Z = np.zeros((max_t, max_x))
        for i, t_x in enumerate(np.arange(max_t)):
            for j, x in enumerate(np.arange(max_x)):
                Z[i, j] = self.conditional_expected_number_of_purchases_up_to_time(t, x, t_x, max_t)

        interpolation = kwargs.pop('interpolation', 'none')

        ax = plt.subplot(111)
        ax.imshow(Z, interpolation=interpolation, **kwargs)
        plt.xlabel("Customer's Historical Frequency")
        plt.ylabel("Customer's Recency")
        plt.title('Expected Number of Future Purchases over 1 Unit of Time,\nby Frequency and Recency of a Customer')

        # turn matrix into square
        forceAspect(ax)

        # necessary for colorbar to show up
        PCM = ax.get_children()[2]
        plt.colorbar(PCM, ax=ax)

        return ax

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
