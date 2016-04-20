import math

from estimation import BetaGeoFitter
import numpy as np
import pandas as pd
import generate_data as gen
import random


class Model(object):
    def __init__(self, N, t):
        """

        Args:
            N:      number of users you're referring to
            t:      time horizon you're looking at
        Returns:

        """
        super(Model, self).__init__()

        if N < 100:
            raise ValueError("Number of users must be statistically relevant. At least 100.")
        if t < 0:
            raise ValueError("Time horizon must be positive.")

        self.t = t
        self.N = N
        self.fitted_model = None
        self.params, self.params_C = None, None
        self.sampled_parameters = None  # result of a bootstrap
        self.numerical_metrics = None

    def fit(self, frequency, recency, T):
        """
        Fit the model to data, finding parameters and their errors, and assigning them to internal variables
        Args:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
        """
        raise NotImplementedError()

    def simulate(self):
        """
        Runs a heavy simulation starting from parameters and their uncertainties,
        to calculate common metrics and their uncertainties.
        """
        raise NotImplementedError()


class BetaGeoModel(Model):
    """
    Fits a BetaGeoModel to the data, and computes relevant metrics by mean of a simulation.
    """

    def fit(self, frequency, recency, T):  # TODO: test it
        bgf = BetaGeoFitter()
        bgf.fit(frequency, recency, T)

        self.params = bgf.params_

        data = pd.DataFrame({'frequency': frequency, 'recency': recency, 'T': T})
        self.estimate_uncertainties_with_bootstrap(data)

        self.fitted_model = bgf

    def estimate_uncertainties_with_bootstrap(self, data, size=10):
        """
        Calculate parameter covariance Matrix by bootstrapping trainig data.

        Args:
            data:   pandas data farme containing 3 columns labelled 'frequency' 'recency' 'T'
            size:   nunmber of re-samplings

        Returns:    The estimated covariance matrix
        """

        if not all(column in data.columns for column in ['frequency', 'recency', 'T']):
            raise ValueError("given data do not contain the 3 magic columns.")
        if size < 2:
            raise ValueError("Run at least 2 samplings to get a covariance.")

        par_estimates = []

        for i in range(size):
            N = len(data)
            sampled_data = data.sample(N, replace=True)

            bgf = BetaGeoFitter(penalizer_coef=0.0)
            bgf.fit(sampled_data['frequency'], sampled_data['recency'], sampled_data['T'])
            par_estimates.append(bgf.params_)

        rs = [par['r'] for par in par_estimates]
        alphas = [par['alpha'] for par in par_estimates]
        As = [par['a'] for par in par_estimates]
        Bs = [par['b'] for par in par_estimates]

        np.cov(rs, alphas, As, Bs)
        x = np.vstack([rs, alphas, As, Bs])
        cov = np.cov(x)
        self.params_C = cov
        self.sampled_parameters = par_estimates

    def evaluate_metrics_with_uncertainties(self, N_syst=10, max_x=10):  # TODO: test it
        """
        Args:
            N_syst:        Number of re-sampling of parameters, for systematic errors
            max_x:         Maximum number of transactions you want to consider
        """

        if self.params is None or self.params_C is None or self.sampled_parameters is None:
            raise ValueError("Model has not been fit yet. Please call the .fit method first.")

        xs = range(max_x)

        measurements_fx = {}
        for x in xs:
            measurements_fx[x] = []
        p_x = [None] * max_x
        p_x_err = [None] * max_x

        for it in range(N_syst):
            par_s = self.sampled_parameters[
                random.randint(0, len(self.sampled_parameters) - 1)]  # pick up a random outcome of the fit
            r_s = par_s['r']
            alpha_s = par_s['alpha']
            a_s = par_s['a']
            b_s = par_s['b']

            data = gen.beta_geometric_nbd_model(self.t, r_s, alpha_s, a_s, b_s, size=self.N)
            n = len(data)
            for x in xs:
                if x == max_x - 1:
                    n_success = len(data[data['frequency'] >= x])  # the last bin is cumulative
                else:
                    n_success = len(data[data['frequency'] == x])
                measurements_fx[x].append(float(n_success) / n)

        for x in xs:
            p_x_err[x] = np.std(measurements_fx[x])  # contain statistic + systematic errors, entangled
            p_x[x] = np.mean(measurements_fx[x])

        self.numerical_metrics = NumericalMetrics(p_x, p_x_err)

    def simulate(self):
        self.evaluate_metrics_with_uncertainties()


class NumericalMetrics(object):
    """
    Contains the metrics common to all transaction counting models (Pareto/NBS, BG/NBD)
    """

    # TODO: add y, Ey, p to common metrics

    def __init__(self, p_x, p_x_err):
        """
        Args:
            p_x:    Probabilities of x (x being the number of transaction per user 0, 1, ...)
            p_x_err:    Error on probabilities of x (x being the number of transaction per user 0, 1, ...)
        """
        super(NumericalMetrics, self).__init__()

        if len(p_x) != len(p_x_err):
            raise ValueError("p_x and p_x_err must have the same length.")

        self.p_x = p_x
        self.p_x_err = p_x_err

    def length(self):
        return len(self.p_x)

    def dump(self):
        print "range: " + str(range(len(self.p_x)))
        print "probabilities: " + str(self.p_x)
        print "probabilities err: " + str(self.p_x_err)

    def expected_x(self):
        """
        Returns:    The E[x] and error as tuple
        """
        Ex = 0
        Ex_err = 0
        for x in range(self.length()):
            Ex += x * self.p_x[x]
            Ex_err += (x * self.p_x_err[x]) ** 2
        return Ex, math.sqrt(Ex_err)


def extract_frequencies(data, max_x=10):
    """
    Given a data frame containing a 'frequency' column, extract multinomial frequencies of purchasing users.
    Args:
        data:   pandas DataFrame containing a 'frequency' column
        max_x:  the maximum x value (number of purchases) to evaluate, the last bin is cumulative, contains all that follow

    Returns:    The frequencies, as list

    """
    fx = []
    n = len(data)
    for x in range(max_x):
        if x == max_x - 1:
            n_success = len(data[data['frequency'] >= x])  # the last bin is cumulative
        else:
            n_success = len(data[data['frequency'] == x])
        fx.append(float(n_success) / n)
    return fx
