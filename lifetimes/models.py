import math
from estimation import BetaGeoFitter, ModifiedBetaGeoFitter, ParetoNBDFitter, BGBBFitter, BGBBBGExtFitter, BGFitter
import numpy as np
import pandas as pd
import generate_data as gen
import random
import copy
from abc import abstractmethod
import uncertainties
from lifetimes.utils import is_almost_equal


class Model(object):
    """
    Base class to handle fitting of a model to data and bootstrap of the parameters.
    """

    def __init__(self):
        self.fitter = None
        self.param_names = None
        self.params, self.params_C = None, None
        self.sampled_parameters = None  # result of a bootstrap
        self.uparams = None

    def is_ready(self):
        return self.params is not None and  self.params_C is not None

    def good_fit(self):
        """
        Decides if we have a good fit
        :return:    bool
        """
        pars = self.params  # it's a dictionary
        if pars is None:
            raise ValueError("fit the data first")

        if not np.all(np.array(pars.values()) >= 0):
            return False
        if not np.all(np.array(pars.values()) <= 500):
            return False
        return True

    def fit(self, frequency, recency, T, bootstrap_size=10, N=None, initial_params=None, iterative_fitting=0):
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
                sampled_N = np.random.multinomial(N_sum, prob, size=1)
                tmp_fitter.fit(frequency=data['frequency'], recency=data['recency'], T=data['T'], N=sampled_N[0])
            par_estimates.append(tmp_fitter.params_)

        par_lists = []
        for par_name in self.param_names:
            par_lists.append([par[par_name] for par in par_estimates])

        par_lists = remove_outliers_from_fitted_params(par_lists)

        x = np.vstack(par_lists)
        cov = np.cov(x)

        self.sampled_parameters = par_estimates
        self.set_parameters(self.params, cov)

    def set_parameters(self, pars, cov):
        """
        Sets patameters and their covariance matrix
        Args:
            pars:   dictionary
            cov:    np.cov
        """
        if len(pars) < len(self.param_names):
            raise ValueError("Provide all parameters!")
        if len(cov) < len(self.param_names) or len(cov[0]) < len(self.param_names):
            raise ValueError("Wrong dimensions of covariance matrix")
        self.params_C = cov
        self.params = {}
        for par_name in self.param_names:
            self.params[par_name] = pars[par_name]

        # set uparams once for all
        if self.param_names is None or self.params is None or self.params_C is None:
            return None
        par_values = uncertainties.correlated_values([self.params[par_name] for par_name in self.param_names],
                                                     self.params_C)
        self.uparams = {}
        for i in range(len(self.param_names)):
            self.uparams[self.param_names[i]] = par_values[i]

    def evaluate_metrics_with_simulation(self, N, t, N_sim=10, max_x=10, tag='frequency'):
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

    def __init__(self, penalizer_coef=0.):
        super(BetaGeoModel, self).__init__()
        self.fitter = BetaGeoFitter(penalizer_coef)
        self.param_names = ['r', 'alpha', 'a', 'b']

    def generateData(self, t, parameters, size):
        return gen.beta_geometric_nbd_model(t, parameters['r'], parameters['alpha'], parameters['a'], parameters['b'],
                                            size)


class ModifiedBetaGeoModel(Model):
    """
    Fits a ModifiedBetaGeoModel to the data, and computes relevant metrics by mean of a simulation.
    """

    def __init__(self, penalizer_coef=0.):
        super(ModifiedBetaGeoModel, self).__init__()
        self.fitter = ModifiedBetaGeoFitter(penalizer_coef)
        self.param_names = ['r', 'alpha', 'a', 'b']

    def generateData(self, t, parameters, size):
        return gen.modified_beta_geometric_nbd_model(t, parameters['r'], parameters['alpha'], parameters['a'],
                                                     parameters['b'],
                                                     size)


class ParetoNBDModel(Model):
    """
    Fits a ParetoNBDModel to the data, and computes relevant metrics by mean of a simulation.
    """

    def __init__(self, penalizer_coef=0.):
        super(ParetoNBDModel, self).__init__()
        self.fitter = ParetoNBDFitter(penalizer_coef)
        self.param_names = ['r', 'alpha', 's', 'beta']
        self.wrapped_static_expected_number_of_purchases_up_to_time = \
            uncertainties.wrap(ParetoNBDFitter.static_expected_number_of_purchases_up_to_time)


    def generateData(self, t, parameters, size):
        return gen.pareto_nbd_model(t, parameters['r'], parameters['alpha'], parameters['s'],
                                    parameters['beta'],
                                    size)

    def expected_number_of_purchases_up_to_time(self, t):
        """
        Args:
            t: a scalar or array of times

        Returns:
            a tuple of two elements: the first is the expected value (or an array of them) and the second is the error
            associated to it (or an array of them)
        """
        if not self.is_ready():
            raise ValueError("Model is not ready. Please call the '.fit' method first or provide parameters.")

        uparams = self.uparams
        r, a, s, b = [uparams[par_name] for par_name in self.param_names]
        return self.wrapped_static_expected_number_of_purchases_up_to_time(r, a, s, b, t)


class BGBBModel(Model):
    """
    Fits a discrete-time BGBB to the data, and computes relevant metrics by mean of a simulation.
    """

    def __init__(self, penalizer_coef=0.):
        super(BGBBModel, self).__init__()
        self.fitter = BGBBFitter(penalizer_coef)
        self.param_names = ['alpha', 'beta', 'gamma', 'delta']
        self.wrapped_static_expected_number_of_purchases_up_to_time = \
            uncertainties.wrap(BGBBFitter.static_expected_number_of_purchases_up_to_time)
        self.wrapped_static_probability_of_n_purchases_up_to_time = \
            uncertainties.wrap(BGBBFitter.static_probability_of_n_purchases_up_to_time)

    def generateData(self, t, parameters, size):
        return gen.bgbb_model(t, parameters['alpha'],
                              parameters['beta'],
                              parameters['gamma'],
                              parameters['delta'],
                              size)

    def expected_number_of_purchases_up_to_time(self, t):
        if not self.is_ready():
            raise ValueError("Model is not ready. Please call the '.fit' method first or provide parameters.")

        uparams = self.uparams
        a, b, g, d = [uparams[par_name] for par_name in self.param_names]
        return self.wrapped_static_expected_number_of_purchases_up_to_time(a, b, g, d, t)

    def probability_of_n_purchases_up_to_time(self, t, n):
        if not self.is_ready():
            raise ValueError("Model is not ready. Please call the '.fit' method first or provide parameters.")

        uparams = self.uparams
        a, b, g, d = [uparams[par_name] for par_name in self.param_names]
        return self.wrapped_static_probability_of_n_purchases_up_to_time(a, b, g, d, t, n)


class BGBBBGExtModel(Model):
    """
       Fits a discrete-time BGBBBG to the data, and computes relevant metrics by mean of a simulation.
     """

    def __init__(self, penalizer_coef=0.):
        super(BGBBBGExtModel, self).__init__()
        self.fitter = BGBBBGExtFitter(penalizer_coef)
        self.param_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'c0']
        self.wrapped_static_expected_number_of_sessions_up_to_time = \
            uncertainties.wrap(BGBBBGExtFitter.static_expected_number_of_sessions_up_to_time)
        self.wrapped_static_probability_of_n_sessions_up_to_time = \
            uncertainties.wrap(BGBBBGExtFitter.static_probability_of_n_sessions_up_to_time)
        self.wrapped_static_expected_probability_of_converting_at_time = \
            uncertainties.wrap(BGBBBGExtFitter.static_regularized_expected_probability_of_converting_at_time)

    def corrected_wrapped_static_expected_probability_of_converting_at_time(self, a, b, g, d, e, z, c0, t):
        uvalue = self.wrapped_static_expected_probability_of_converting_at_time(a, b, g, d, e, z, c0, t)
        if math.isnan(uvalue.n) or uvalue.n > 1.0 or uvalue.n < 0.0:
            uvalue = uncertainties.ufloat(0.0, 0.0)
        if math.isnan(uvalue.s) or uvalue.s > 1.0:
            uvalue = uncertainties.ufloat(uvalue.n, 0.0)
        return uvalue

    def generateData(self, t, parameters, size):
        return gen.bgbbbgext_model(t, parameters['alpha'],
                                   parameters['beta'],
                                   parameters['gamma'],
                                   parameters['delta'],
                                   parameters['epsilon'],
                                   parameters['zeta'],
                                   parameters['c0'],
                                   size)

    def fit(self, frequency, recency, T, bootstrap_size=10, N=None, initial_params=None,
            iterative_fitting=0, frequency_before_conversion=None):
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

        self.fitter.fit(frequency=frequency, recency=recency, T=T,
                        frequency_before_conversion=frequency_before_conversion, N=N,
                        initial_params=initial_params,
                        iterative_fitting=iterative_fitting)

        self.params = self.fitter.params_

        if N is None:
            data = pd.DataFrame(
                {'frequency': frequency, 'recency': recency, 'T': T,
                 'frequency_before_conversion': frequency_before_conversion})
            self._estimate_uncertainties_with_bootstrap(data, bootstrap_size)
        else:
            data = pd.DataFrame(
                {'frequency': frequency, 'recency': recency, 'T': T,
                 'frequency_before_conversion': frequency_before_conversion, 'N': N})
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

        self.sampled_parameters = par_estimates
        self.set_parameters(self.params, cov)

    def expected_number_of_sessions_up_to_time(self, t):
        if not self.is_ready():
            raise ValueError("Model is not ready. Please call the '.fit' method first or provide parameters.")

        uparams = self.uparams
        a, b, g, d, e, z, c0 = [uparams[par_name] for par_name in self.param_names]
        return self.wrapped_static_expected_number_of_sessions_up_to_time(a, b, g, d, t)

    def probability_of_n_sessions_up_to_time(self, t, n):
        if not self.is_ready():
            raise ValueError("Model is not ready. Please call the '.fit' method first or provide parameters.")

        uparams = self.uparams
        a, b, g, d, e, z, c0 = [uparams[par_name] for par_name in self.param_names]
        return self.wrapped_static_probability_of_n_sessions_up_to_time(a, b, g, d, t, n)

    def expected_probability_of_converting_at_time(self, t):
        if not self.is_ready():
            raise ValueError("Model is not ready. Please call the '.fit' method first or provide parameters.")

        uparams = self.uparams
        a, b, g, d, e, z, c0 = [uparams[par_name] for par_name in self.param_names]
        return self.corrected_wrapped_static_expected_probability_of_converting_at_time(a, b, g, d, e, z, c0, t)

    def expected_probability_of_converting_within_time(self, t):  #TODO: unstable.. fix it
        if not self.is_ready():
            raise ValueError("Model is not ready. Please call the '.fit' method first or provide parameters.")

        res = 0.0
        for ti in range(t + 1):
            res += self.expected_probability_of_converting_at_time(ti)
        return res


class BGModel(Model):
    """
    Fits a discrete-time BG to the data, and computes relevant metrics by mean of a simulation.
    """

    def __init__(self, penalizer_coef=0.):
        super(BGModel, self).__init__()
        self.fitter = BGFitter(penalizer_coef)
        self.param_names = ['alpha', 'beta']
        self.params, self.params_C = None, None
        self.sampled_parameters = None  # result of a bootstrap
        self.uparams = None
        self.wrapped_static_expected_number_of_purchases_up_to_time = \
            uncertainties.wrap(BGFitter.static_expected_number_of_purchases_up_to_time)
        self.wrapped_static_probability_of_n_purchases_up_to_time = \
            uncertainties.wrap(BGFitter.static_probability_of_n_purchases_up_to_time)

    def generate_data(self, t, parameters, size):
        return gen.bgext_model(t, parameters['alpha'],
                               parameters['beta'],
                               size=size)

    def fit(self, frequency, T, recency=None,  bootstrap_size=10, N=None, initial_params=None, iterative_fitting=0):
        """
        Fit the model to data, finding parameters and their errors, and assigning them to internal variables
        Args:
            frequency: the frequency vector of customers' sessions (denoted x in literature).
            T: the vector of customers' age (time since first purchase)
            recency:    None and useless, inserted just the keep the inheritance from Model
            bootstrap_size: number of data-samplings used to address parameter uncertainty
            N:  count of users matching FRT (compressed data), if absent data are assumed to be non-compressed
        """

        self.fitter.fit(frequency=frequency, T=T, N=N,
                        initial_params=initial_params,
                        iterative_fitting=iterative_fitting)

        self.params = self.fitter.params_

        if N is None:
            data = pd.DataFrame({'frequency': frequency, 'T': T})
            self._estimate_uncertainties_with_bootstrap(data, bootstrap_size)
        else:
            data = pd.DataFrame({'frequency': frequency, 'T': T, 'N': N})
            self._estimate_uncertainties_with_bootstrap(data, bootstrap_size, compressed_data=True)

    def _estimate_uncertainties_with_bootstrap(self, data, size=10, compressed_data=False):

        if not all(column in data.columns for column in ['frequency', 'T']):
            raise ValueError("given data do not contain the 2 magic columns.")
        if size < 2:
            raise ValueError("Run at least 2 samplings to get a covariance.")

        par_estimates = []

        tmp_fitter = copy.deepcopy(self.fitter)
        for i in range(size):
            N = len(data)
            if compressed_data is False:
                sampled_data = data.sample(N, replace=True)
                tmp_fitter.fit(sampled_data['frequency'], sampled_data['T'])
            else:
                # in case of compressed data you've gotta sample a multinomial distribution
                N = data['N']
                N_sum = sum(N)
                prob = [float(n) / N_sum for n in N]
                sampled_N = np.random.multinomial(N_sum, prob, size=1)
                tmp_fitter.fit(frequency=data['frequency'], T=data['T'], N=sampled_N[0])
            par_estimates.append(tmp_fitter.params_)

        par_lists = []
        for par_name in self.param_names:
            par_lists.append([par[par_name] for par in par_estimates])

        par_lists = remove_outliers_from_fitted_params(par_lists)
        x = np.vstack(par_lists)
        cov = np.cov(x)

        self.sampled_parameters = par_estimates
        self.set_parameters(self.params, cov)

    def expected_number_of_purchases_up_to_time(self, t):
        uparams = self.uparams
        a, b = [uparams[par_name] for par_name in self.param_names]
        return self.wrapped_static_expected_number_of_purchases_up_to_time(a, b, t)

    def probability_of_n_purchases_up_to_time(self, t, n):
        uparams = self.uparams
        a, b = [uparams[par_name] for par_name in self.param_names]
        return self.wrapped_static_probability_of_n_purchases_up_to_time(a, b, t, n)


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


def remove_outliers_from_fitted_params(par_lists, method='Gaussian'):
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
    diff = (points - median) ** 2
    diff = np.array(np.sqrt(diff))
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return ((modified_z_score > thresh) * 1 + (points < 0) * 1) > 0
