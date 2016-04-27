import math
from estimation import BetaGeoFitter, ModifiedBetaGeoFitter, ParetoNBDFitter
import numpy as np
import pandas as pd
import generate_data as gen
import random
import copy
from abc import ABCMeta, abstractmethod


class Model(object): # , metaclass=ABCMeta):
    """
    Base class to handle fitting of a model to data and bootstrap of the parameters.
    """

    def __init__(self):
        self.fitter = None
        self.param_names = None
        self.params, self.params_C = None, None
        self.sampled_parameters = None  # result of a bootstrap

    def fit(self, frequency, recency, T, N = None, bootstrap_size=10):
        """
        Fit the model to data, finding parameters and their errors, and assigning them to internal variables
        Args:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            bootstrap_size: number of data-samplings used to address parameter uncertainty
        """
        pass
        self.fitter.fit(frequency, recency, T, N)

        self.params = self.fitter.params_

        if N is None:
            data = pd.DataFrame({'frequency': frequency, 'recency': recency, 'T': T})
            self.estimate_uncertainties_with_bootstrap(data, bootstrap_size)
        else:
            data = pd.DataFrame({'frequency': frequency, 'recency': recency, 'T': T, 'N': N})
            self.estimate_uncertainties_with_bootstrap(data, bootstrap_size, compressed_data=True)


    @abstractmethod
    def generateData(self, t, parameters, size):
        """
        Generate a dataset with from the current model
        Args:
            t: purchase time horizon
            parameters: dictionary with keys r,alpha,a,b and parameter values
            size: number of samples to generate
        Returns:    dataframe with keys 'recency','frequency','T'
        """
        pass

    def estimate_uncertainties_with_bootstrap(self, data, size=10, compressed_data = False):
        """
        Calculate parameter covariance Matrix by bootstrapping trainig data.

        Args:
            data:   pandas data farme containing 3 columns labelled 'frequency' 'recency' 'T'
            size:   number of re-samplings
        """

        if not all(column in data.columns for column in ['frequency', 'recency', 'T']):
            raise ValueError("given data do not contain the 3 magic columns.")
        if size < 2:
            raise ValueError("Run at least 2 samplings to get a covariance.")

        par_estimates = []

        tmp_fitter = copy.deepcopy(self.fitter)
        for i in range(size):
            N = len(data)
            if compressed_data is False:
                sampled_data = data.sample(N, replace=True)
                tmp_fitter.fit(sampled_data['frequency'], sampled_data['recency'], sampled_data['T'])
            else:
                # in case of compressed data you've gotta sample a multinomial distribution # TODO: test
                N = data['N']
                N_sum = sum(N)
                prob = [float(n)/N_sum for n in N]
                sampled_N = np.random.multinomial(N_sum, prob, size=1)
                tmp_fitter.fit(data['frequency'], data['recency'], data['T'], sampled_N)
            par_estimates.append(tmp_fitter.params_)

        par_lists = []
        for par_name in self.param_names:
            par_lists.append([par[par_name] for par in par_estimates])

        x = np.vstack(par_lists)
        cov = np.cov(x)
        self.params_C = cov
        self.sampled_parameters = par_estimates

    def evaluate_metrics_with_simulation(self, N, t, N_sim=10, max_x=10):  # TODO: test it
        """
        Args:
            N:      number of users you're referring to
            t:      time horizon you're looking at
            N_sim:        Number of simulations
            max_x:         Maximum number of transactions you want to consider
        Returns:    The numerical metrics
        """

        if self.params is None or self.params_C is None or self.sampled_parameters is None:
            raise ValueError("Model has not been fit yet. Please call the '.fit' method first.")

        xs = range(max_x)

        frequencies = {}
        for x in xs:
            frequencies[x] = []
        p_x = [None] * max_x
        p_x_err = [None] * max_x

        for sim_i in range(N_sim):
            par_s = self.sampled_parameters[
                random.randint(0, len(self.sampled_parameters) - 1)]  # pick up a random outcome of the fit
            data = self.generateData(t, par_s, N)
            n = len(data)
            for x in xs:
                if x == max_x - 1:
                    n_success = len(data[data['frequency'] >= x])  # the last bin is cumulative
                else:
                    n_success = len(data[data['frequency'] == x])
                frequencies[x].append(float(n_success) / n)

        for x in xs:
            p_x_err[x] = np.std(frequencies[x])  # contain statistic + systematic errors, entangled
            p_x[x] = np.mean(frequencies[x])

        return NumericalMetrics(p_x, p_x_err)


class BetaGeoModel(Model):
    """
    Fits a BetaGeoModel to the data, and computes relevant metrics by mean of a simulation.
    """

    def __init__(self):
        super(BetaGeoModel, self).__init__()
        self.fitter = BetaGeoFitter()
        self.param_names = ['r', 'alpha', 'a', 'b']

    def generateData(self, t, parameters, size):
        return gen.beta_geometric_nbd_model(t, parameters['r'], parameters['alpha'], parameters['a'], parameters['b'],
                                            size)


class ModifiedBetaGeoModel(Model):
    """
    Fits a ModifiedBetaGeoModel to the data, and computes relevant metrics by mean of a simulation.
    """

    def __init__(self):

        super(ModifiedBetaGeoModel, self).__init__()
        self.fitter = ModifiedBetaGeoFitter()
        self.param_names = ['r', 'alpha', 'a', 'b']

    def generateData(self, t, parameters, size):
        return gen.modified_beta_geometric_nbd_model(t, parameters['r'], parameters['alpha'], parameters['a'],
                                                     parameters['b'],
                                                     size)


class ParetoNBDModel(Model):
    """
    Fits a ParetoNBDModel to the data, and computes relevant metrics by mean of a simulation.
    """

    def __init__(self):
        super(ParetoNBDModel, self).__init__()
        self.fitter = ParetoNBDFitter()
        self.param_names = ['r', 'alpha', 's', 'beta']

    def generateData(self, t, parameters, size):
        return gen.pareto_nbd_model(t, parameters['r'], parameters['alpha'], parameters['s'],
                                    parameters['beta'],
                                    size)


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
    if 'frequency' not in data.columns:
        raise ValueError("data does not contain frequency column")

    fx = []
    n = len(data)
    for x in range(max_x):
        if x == max_x - 1:
            n_success = len(data[data['frequency'] >= x])  # the last bin is cumulative
        else:
            n_success = len(data[data['frequency'] == x])
        fx.append(float(n_success) / n)
    return fx
