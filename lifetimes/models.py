import math
from estimation import BetaGeoFitter, ModifiedBetaGeoFitter, ParetoNBDFitter, BGBBFitter, BGBBBBFitter, BGBBBGFitter
import numpy as np
import pandas as pd
import generate_data as gen
import random
import copy
from abc import abstractmethod


class Model(object):
    """
    Base class to handle fitting of a model to data and bootstrap of the parameters.
    """

    def __init__(self):
        self.fitter = None
        self.param_names = None
        self.params, self.params_C = None, None
        self.sampled_parameters = None  # result of a bootstrap

    def fit(self, frequency, recency, T, bootstrap_size=10, N=None, initial_params=None, iterative_fitting=1):
        """
        Fit the model to data, finding parameters and their errors, and assigning them to internal variables
        Args:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            bootstrap_size: number of data-samplings used to address parameter uncertainty
            N:  count of users matching FRT (compressed data), if absent data are assumed to be non-compressed
        """
        self.fitter.fit(frequency=frequency, recency=recency, T=T, N=N, initial_params=initial_params,
                        iterative_fitting=iterative_fitting)

        self.params = self.fitter.params_

        if N is None:
            data = pd.DataFrame({'frequency': frequency, 'recency': recency, 'T': T})
            self._estimate_uncertainties_with_bootstrap(data, bootstrap_size)
        else:
            data = pd.DataFrame({'frequency': frequency, 'recency': recency, 'T': T, 'N': N})
            self._estimate_uncertainties_with_bootstrap(data, bootstrap_size, compressed_data=True)

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

    def _estimate_uncertainties_with_bootstrap(self, data, size=10, compressed_data=False):
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
                # in case of compressed data you've gotta sample a multinomial distribution
                N = data['N']
                N_sum = sum(N)
                prob = [float(n) / N_sum for n in N]
                sampled_N = np.random.multinomial(N_sum, prob, size = 1)
                tmp_fitter.fit(frequency=data['frequency'],recency = data['recency'], T = data['T'], N = sampled_N[0])
            par_estimates.append(tmp_fitter.params_)

        par_lists = []
        for par_name in self.param_names:
            par_lists.append([par[par_name] for par in par_estimates])

        par_lists = remove_outliers_from_fitted_params(par_lists)

        x = np.vstack(par_lists)
        cov = np.cov(x)
        self.params_C = cov
        self.sampled_parameters = par_estimates

    def evaluate_metrics_with_simulation(self, N, t, N_sim=10, max_x=10, tag = 'frequency'):
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
            if tag not in data:
                raise ValueError("Unreconized column: " + tag)
            n = len(data)
            for x in xs:
                if x == max_x - 1:
                    n_success = len(data[data[tag] >= x])  # the last bin is cumulative
                else:
                    n_success = len(data[data[tag] == x])
                frequencies[x].append(float(n_success) / n)

        for x in xs:
            p_x_err[x] = np.std(frequencies[x])  # contain statistic + systematic errors, entangled
            p_x[x] = np.mean(frequencies[x])

        return NumericalMetrics(p_x, p_x_err)

    def parameters_dictionary_from_list(self, parameters):
        """

        Args:
            parameters: a plain list containing the parameters

        Returns:
            a dictionary containing the parameters to be used by the model
        """
        if len(self.param_names) != len(parameters):
            raise ValueError("wrong number of parameter passed")

        param_dictionary = {}
        for parameter, name in zip(parameters, self.param_names):
            param_dictionary[name] = parameter
        return param_dictionary


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

    def expected_number_of_purchases_up_to_time_with_errors(self, t):
        """

        Args:
            t: a scalar or array of times

        Returns:
            a tuple of two elements: the first is the expected value (or an array of them) and the second is the error
            associated to it (or an array of them)
        """
        if self.params is None or self.params_C is None:
            raise ValueError("Model has not been fit yet. Please call the '.fit' method first.")

        return self.fitter.expected_number_of_purchases_up_to_time(t), \
               self.fitter.expected_number_of_purchases_up_to_time_error(t, self.params_C)


class BGBBModel(Model):
    """
    Fits a discrete-time BGBB to the data, and computes relevant metrics by mean of a simulation.
    """

    def __init__(self):
        super(BGBBModel, self).__init__()
        self.fitter = BGBBFitter()
        self.param_names = ['alpha', 'beta', 'gamma', 'delta']

    def generateData(self, t, parameters, size):
        return gen.bgbb_model(t, parameters['alpha'],
                              parameters['beta'],
                              parameters['gamma'],
                              parameters['delta'],
                              size)

    def expected_number_of_purchases_up_to_time_with_errors(self, t):
        """

        Args:
            t: a scalar or array of times

        Returns:
            a tuple of two elements: the first is the expected value (or an array of them) and the second is the error
            associated to it (or an array of them)
        """
        if self.params is None or self.params_C is None:
            raise ValueError("Model has not been fit yet. Please call the '.fit' method first.")

        return self.fitter.expected_number_of_purchases_up_to_time(
            t), self.fitter.expected_number_of_purchases_up_to_time_error(t, self.params_C)


class BGBBBGModel(Model):
    """
       Fits a discrete-time BGBBBG to the data, and computes relevant metrics by mean of a simulation.
     """

    def __init__(self):
        super(BGBBBGModel, self).__init__()
        self.fitter = BGBBBGFitter()
        self.param_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']

    def generateData(self, t, parameters, size):
        return gen.bgbbbg_model(t, parameters['alpha'],
                                parameters['beta'],
                                parameters['gamma'],
                                parameters['delta'],
                                parameters['epsilon'],
                                parameters['zeta'],
                                size)

    def fit(self, frequency, recency, T, bootstrap_size=10, N=None, initial_params=None,
            iterative_fitting=1, frequency_before_conversion=None):
        """
        Fit the model to data, finding parameters and their errors, and assigning them to internal variables
        Args:
            frequency: the frequency vector of customers' sessions (denoted x in literature).
            recency: the recency vector of customers' sessions (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            bootstrap_size: number of data-samplings used to address parameter uncertainty
            N:  count of users matching FRT (compressed data), if absent data are assumed to be non-compressed
            frequency_before_conversion:  the frequency vector of customers' sessions before first purchase--> Must be a valid array
        """

        if frequency_before_conversion is None:
            raise ValueError("You must provide a valid vector of frequency_before_purchase")

        self.fitter.fit(frequency=frequency, recency=recency, T=T, frequency_before_conversion=frequency_before_conversion, N=N,
                        initial_params=initial_params,
                        iterative_fitting=iterative_fitting)

        self.params = self.fitter.params_

        if N is None:
            data = pd.DataFrame(
                {'frequency': frequency, 'recency': recency, 'T': T, 'frequency_before_conversion': frequency_before_conversion})
            self._estimate_uncertainties_with_bootstrap(data, bootstrap_size)
        else:
            data = pd.DataFrame(
                {'frequency': frequency, 'recency': recency, 'T': T, 'frequency_before_conversion': frequency_before_conversion, 'N': N})
            self._estimate_uncertainties_with_bootstrap(data, bootstrap_size, compressed_data=True)

    def _estimate_uncertainties_with_bootstrap(self, data, size=10, compressed_data=False):

        if not all(column in data.columns for column in ['frequency', 'recency', 'T', 'frequency_before_conversion']):
            raise ValueError("given data do not contain the 4 magic columns.")
        if size < 2:
            raise ValueError("Run at least 2 samplings to get a covariance.")

        par_estimates = []

        tmp_fitter = copy.deepcopy(self.fitter)
        for i in range(size):
            N = len(data)
            if compressed_data is False:
                sampled_data = data.sample(N, replace=True)
                tmp_fitter.fit(sampled_data['frequency'], sampled_data['recency'], sampled_data['T'],
                               sampled_data['frequency_before_conversion'])
            else:
                # in case of compressed data you've gotta sample a multinomial distribution
                N = data['N']
                N_sum = sum(N)
                prob = [float(n) / N_sum for n in N]
                sampled_N = np.random.multinomial(N_sum, prob, size=1)
                tmp_fitter.fit(frequency=data['frequency'], recency=data['recency'], T=data['T'],
                               frequency_before_conversion=data['frequency_before_conversion'], N=sampled_N[0])
            par_estimates.append(tmp_fitter.params_)

        par_lists = []
        for par_name in self.param_names:
            par_lists.append([par[par_name] for par in par_estimates])

        par_lists = remove_outliers_from_fitted_params(par_lists)
        x = np.vstack(par_lists)
        cov = np.cov(x)
        self.params_C = cov
        self.sampled_parameters = par_estimates
        self.par_lists = par_lists

    def expected_probability_of_converting_at_time_with_error(self, t):
        value = self.fitter.expected_probability_of_converting_at_time(t)
        error = self.fitter.expected_probability_of_converting_at_time_error(t, zip(*self.par_lists))
        return value, error


class BGBBBBModel(Model):
    """
    Fits a discrete-time BGBBBB to the data, and computes relevant metrics by mean of a simulation.
    """

    def __init__(self):
        super(BGBBBBModel, self).__init__()
        self.fitter = BGBBBBFitter()
        self.param_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']

    def generateData(self, t, parameters, size):
        return gen.bgbbbb_model(t, parameters['alpha'],
                                parameters['beta'],
                                parameters['gamma'],
                                parameters['delta'],
                                parameters['epsilon'],
                                parameters['zeta'],
                                size)

    def fit(self, frequency, recency, T, bootstrap_size=10, N=None, initial_params=None,
            iterative_fitting=1, frequency_purchases=None):
        """
        Fit the model to data, finding parameters and their errors, and assigning them to internal variables
        Args:
            frequency: the frequency vector of customers' sessions (denoted x in literature).
            recency: the recency vector of customers' sessions (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            bootstrap_size: number of data-samplings used to address parameter uncertainty
            N:  count of users matching FRT (compressed data), if absent data are assumed to be non-compressed
            frequency_purchases:  the frequency vector of customers' purchases --> Must be a valid array
        """

        if frequency_purchases is None:
            raise ValueError("You must provide a valid vector of frequency_purchases")

        self.fitter.fit(frequency=frequency, recency=recency, T=T, frequency_purchases=frequency_purchases, N=N,
                        initial_params=initial_params,
                        iterative_fitting=iterative_fitting)

        self.params = self.fitter.params_

        if N is None:
            data = pd.DataFrame(
                {'frequency': frequency, 'recency': recency, 'T': T, 'frequency_purchases': frequency_purchases})
            self._estimate_uncertainties_with_bootstrap(data, bootstrap_size)
        else:
            data = pd.DataFrame(
                {'frequency': frequency, 'recency': recency, 'T': T, 'frequency_purchases': frequency_purchases,'N': N})
            self._estimate_uncertainties_with_bootstrap(data, bootstrap_size, compressed_data=True)

    def _estimate_uncertainties_with_bootstrap(self, data, size=10, compressed_data=False):

        if not all(column in data.columns for column in ['frequency', 'recency', 'T', 'frequency_purchases']):
            raise ValueError("given data do not contain the 4 magic columns.")
        if size < 2:
            raise ValueError("Run at least 2 samplings to get a covariance.")

        par_estimates = []

        tmp_fitter = copy.deepcopy(self.fitter)
        for i in range(size):
            N = len(data)
            if compressed_data is False:
                sampled_data = data.sample(N, replace=True)
                tmp_fitter.fit(sampled_data['frequency'], sampled_data['recency'], sampled_data['T'],
                               sampled_data['frequency_purchases'])
            else:
                # in case of compressed data you've gotta sample a multinomial distribution
                N = data['N']
                N_sum = sum(N)
                prob = [float(n) / N_sum for n in N]
                sampled_N = np.random.multinomial(N_sum, prob, size=1)
                tmp_fitter.fit(frequency=data['frequency'],recency=data['recency'], T=data['T'],frequency_purchases = data['frequency_purchases'], N=sampled_N[0])
            par_estimates.append(tmp_fitter.params_)

        par_lists = []
        for par_name in self.param_names:
            par_lists.append([par[par_name] for par in par_estimates])

        par_lists = remove_outliers_from_fitted_params(par_lists)
        x = np.vstack(par_lists)
        cov = np.cov(x)
        self.params_C = cov
        self.sampled_parameters = par_estimates

    def expected_number_of_sessions_up_to_time_with_errors(self, t):
        """

        Args:
            t: a scalar or array of times

        Returns:
            a tuple of two elements: the first is the expected value (or an array of them) and the second is the error
            associated to it (or an array of them)
        """
        if self.params is None or self.params_C is None:
            raise ValueError("Model has not been fit yet. Please call the '.fit' method first.")

        C = self.params_C[0:4, 0:4]  # extract sub-matrix

        return self.fitter.expected_number_of_sessions_up_to_time(
            t), self.fitter.expected_number_of_sessions_up_to_time_error(t, C)

    def expected_number_of_purchases_up_to_time_with_errors(self, t):
        """

        Args:
            t: a scalar or array of times

        Returns:
            a tuple of two elements: the first is the expected value (or an array of them) and the second is the error
            associated to it (or an array of them)
        """
        if self.params is None or self.params_C is None:
            raise ValueError("Model has not been fit yet. Please call the '.fit' method first.")

        return self.fitter.expected_number_of_purchases_up_to_time(
            t), self.fitter.expected_number_of_purchases_up_to_time_error(t, self.params_C)


class NumericalMetrics(object):
    """
    Contains the metrics common to all transaction counting models (Pareto/NBS, BG/NBD, BG/BB)
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


def remove_outliers_from_fitted_params(par_lists, method = 'Gaussian'):
    is_outlier_lists = []
    for par_list in par_lists:
        is_outlier_lists.append(is_outlier(par_list))
    is_outlier_full = [False] * len(is_outlier_lists[0])
    for is_outlier_list in is_outlier_lists:
        is_outlier_full = is_outlier_full | is_outlier_list

    result = []
    for par_list in par_lists:
        new_list = []
        for i in range(len(par_list)):
            if not is_outlier_full[i]:
                new_list.append(par_list[i])
        result.append(new_list)
    return result


def is_outlier(points, thresh=4):
    points = np.array(points)
    median = np.median(points)
    diff = (points - median)**2
    diff = np.array(np.sqrt(diff))
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return ((modified_z_score > thresh)*1 + (points < 0)*1) > 0
