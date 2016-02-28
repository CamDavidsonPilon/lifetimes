from __future__ import print_function
from collections import OrderedDict

import numpy as np
from numpy import log, exp, logaddexp, asarray, any as npany, c_ as vconcat,\
    isinf, isnan, ones_like
from pandas import DataFrame

from scipy import special
from scipy import misc

from lifetimes.utils import _fit, _scale_time, _check_inputs, customer_lifetime_value
from lifetimes.generate_data import pareto_nbd_model, beta_geometric_nbd_model, modified_beta_geometric_nbd_model

__all__ = ['BetaGeoFitter', 'ParetoNBDFitter', 'GammaGammaFitter', 'ModifiedBetaGeoFitter']


class BaseFitter(object):

    def __repr__(self):
        classname = self.__class__.__name__
        try:
            s = """<lifetimes.%s: fitted with %d subjects, %s>""" % (classname, self.data.shape[0], self._print_params())
        except AttributeError:
            s = """<lifetimes.%s>""" % classname
        return s

    def _unload_params(self, *args):
        if not hasattr(self, 'params_'):
            raise ValueError("Model has not been fit yet. Please call the .fit method first.")
        return [self.params_[x] for x in args]

    def _print_params(self):
        s = ""
        for p, value in self.params_.items():
            s += "%s: %.2f, " % (p, value)
        return s.strip(', ')


class GammaGammaFitter(BaseFitter):

    def __init__(self, penalizer_coef=0.):
        self.penalizer_coef = penalizer_coef

    @staticmethod
    def _negative_log_likelihood(params, frequency, avg_monetary_value, penalizer_coef=0):
        if any(i < 0 for i in params):
            return np.inf

        p, q, v = params

        x = frequency
        m = avg_monetary_value

        negative_log_likelihood_values = special.gammaln(p * x + q) - special.gammaln(p * x) - special.gammaln(q)\
            + q * np.log(v) + (p * x - 1) * np.log(m) + (p * x) * np.log(x) - (p * x + q) * np.log(x * m + v)

        penalizer_term = penalizer_coef * log(params).sum()
        negative_log_likelihood = -np.sum(negative_log_likelihood_values) + penalizer_term

        return negative_log_likelihood

    def conditional_expected_average_profit(self, frequency=None, monetary_value=None):
        """
        This method computes the conditional expectation of the average profit per transaction
        for a group of one or more customers.
            x: a vector containing the customers' frequencies. Defaults to the whole set of
                frequencies used for fitting the model.
            m: a vector containing the customers' monetary values. Defaults to the whole set of
                monetary values used for fitting the model.

        Returns:
            the conditional expectation of the average profit per transaction
        """
        m = self.data['monetary_value'] if monetary_value is None else monetary_value
        x = self.data['frequency'] if frequency is None else frequency
        p, q, v = self._unload_params('p', 'q', 'v')
        return (((q - 1) / (p * x + q - 1)) * (v * p / (q - 1))) + (p * x / (p * x + q - 1)) * m

    def fit(self, frequency, monetary_value, iterative_fitting=5, initial_params=None, verbose=False):
        """
        This methods fits the data to the Gamma/Gamma model.

        Parameters:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            monetary_value: the monetary value vector of customer's purchases (denoted m in literature).
            iterative_fitting: perform `iterative_fitting` additional fits to find the best
                parameters for the model. Setting to 0 will improve performances but possibly
                hurt estimates. This model is not very stable so we suggest >10 for best estimates evaluation.
            initial_params: set initial params for the iterative fitter.
            verbose: set to true to print out convergence diagnostics.

        Returns:
            self, fitted and with parameters estimated
        """
        params, self._negative_log_likelihood_ = _fit(self._negative_log_likelihood,
                                                      [frequency, monetary_value, self.penalizer_coef],
                                                      iterative_fitting,
                                                      initial_params,
                                                      3,
                                                      verbose)

        self.data = DataFrame(vconcat[frequency, monetary_value], columns=['frequency', 'monetary_value'])
        self.params_ = OrderedDict(zip(['p', 'q', 'v'], params))

        return self

    def customer_lifetime_value(self, transaction_prediction_model, frequency, recency, T, monetary_value, time=12, discount_rate=1):
        """
        This method computes the average lifetime value for a group of one or more customers.
            transaction_prediction_model: the model to predict future transactions, literature uses
                pareto/ndb but we can also use a different model like bg
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            monetary_value: the monetary value vector of customer's purchases (denoted m in literature).
            time: the lifetime expected for the user in months. Default: 12
            discount_rate: the monthly adjusted discount rate. Default: 1

        Returns:
            Series object with customer ids as index and the estimated customer lifetime values as values
        """
        adjusted_monetary_value = self.conditional_expected_average_profit(frequency, monetary_value)  # use the Gamma-Gamma estimates for the monetary_values
        return customer_lifetime_value(transaction_prediction_model, frequency, recency, T, adjusted_monetary_value, time, discount_rate)


class ParetoNBDFitter(BaseFitter):

    def __init__(self, penalizer_coef=0.):
        self.penalizer_coef = penalizer_coef

    def fit(self, frequency, recency, T, iterative_fitting=1, initial_params=None, verbose=False):
        """
        This methods fits the data to the Pareto/NBD model.

        Parameters:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            iterative_fitting: perform `iterative_fitting` additional fits to find the best
                parameters for the model. Setting to 0 will improve performances but possibly
                hurt estimates.
            initial_params: set initial params for the iterative fitter.
            verbose: set to true to print out convergence diagnostics.

        Returns:
            self, with additional properties and methods like params_ and plot

        """
        frequency = asarray(frequency)
        recency = asarray(recency)
        T = asarray(T)
        _check_inputs(frequency, recency, T)

        params, self._negative_log_likelihood_ = _fit(self._negative_log_likelihood,
                                                      [frequency, recency, T, self.penalizer_coef],
                                                      iterative_fitting,
                                                      initial_params,
                                                      4,
                                                      verbose)

        self.params_ = OrderedDict(zip(['r', 'alpha', 's', 'beta'], params))
        self.data = DataFrame(vconcat[frequency, recency, T], columns=['frequency', 'recency', 'T'])
        self.generate_new_data = lambda size=1: pareto_nbd_model(T, *params, size=size)

        self.predict = self.conditional_expected_number_of_purchases_up_to_time
        return self

    @staticmethod
    def _log_A_0(params, freq, recency, age):

        r, alpha, s, beta = params

        min_of_alpha_beta, max_of_alpha_beta, t = (alpha, beta, r + freq) if alpha < beta else (beta, alpha, s + 1)
        abs_alpha_beta = max_of_alpha_beta - min_of_alpha_beta

        rsf = r + s + freq
        p_1, q_1 = special.hyp2f1(rsf, t, rsf + 1., abs_alpha_beta / (max_of_alpha_beta + recency)), (max_of_alpha_beta + recency)
        p_2, q_2 = special.hyp2f1(rsf, t, rsf + 1., abs_alpha_beta / (max_of_alpha_beta + age)), (max_of_alpha_beta + age)

        try:
            size = len(freq)
            sign = np.ones(size)
        except TypeError:
            sign = 1

        return misc.logsumexp([log(p_1) + rsf * log(q_2), log(p_2) + rsf * log(q_1)], axis=0, b=[sign, -sign]) \
            - rsf * log(q_1 * q_2)

    @staticmethod
    def _negative_log_likelihood(params, freq, rec, T, penalizer_coef):

        if npany(asarray(params) <= 0.):
            return np.inf

        r, alpha, s, beta = params
        x = freq

        r_s_x = r + s + x

        A_1 = special.gammaln(r + x) - special.gammaln(r) + r * log(alpha) + s * log(beta)
        log_A_0 = ParetoNBDFitter._log_A_0(params, freq, rec, T)

        A_2 = logaddexp(-(r + x) * log(alpha + T) - s * log(beta + T), log(s) + log_A_0 - log(r_s_x))

        penalizer_term = penalizer_coef * log(params).sum()
        return -(A_1 + A_2).sum() + penalizer_term

    def conditional_probability_alive(self, frequency, recency, T):
        """
        Compute the probability that a customer with history (frequency, recency, T) is currently
        alive. From http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf

        Parameters:
            frequency: a scalar: historical frequency of customer.
            recency: a scalar: historical recency of customer.
            T: a scalar: age of the customer.

        Returns: a scalar value representing a probability
        """
        x, t_x = frequency, recency
        r, alpha, s, beta = self._unload_params('r', 'alpha', 's', 'beta')

        A_0 = np.exp(self._log_A_0([r, alpha, s, beta], x, t_x, T))
        return 1. / (1. + (s / (r + s + x)) * (alpha + T) ** (r + x) * (beta + T) ** s * A_0)

    def conditional_probability_alive_matrix(self, max_frequency=None, max_recency=None):
        """
        Compute the probability alive matrix
        Parameters:
            max_frequency: the maximum frequency to plot. Default is max observed frequency.
            max_recency: the maximum recency to plot. This also determines the age of the customer.
                Default to max observed age.

        Returns a matrix of the form [t_x: historical recency, x: historical frequency]

        """

        max_frequency = max_frequency or int(self.data['frequency'].max())
        max_recency = max_recency or int(self.data['T'].max())

        Z = np.zeros((max_recency + 1, max_frequency + 1))
        for i, recency in enumerate(np.arange(max_recency + 1)):
            for j, frequency in enumerate(np.arange(max_frequency + 1)):
                Z[i, j] = self.conditional_probability_alive(frequency, recency, max_recency)

        return Z

    def conditional_expected_number_of_purchases_up_to_time(self, t, frequency, recency, T):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population, given they have purchase history (frequency, recency, T)

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
        first_term = (special.gamma(r + x) / special.gamma(r)) * (alpha ** r * beta ** s) / (alpha + T) ** (r + x) / (beta + T) ** s
        second_term = (r + x) * (beta + T) / (alpha + T) / (s - 1)
        third_term = 1 - ((beta + T) / (beta + T + t)) ** (s - 1)
        return first_term * second_term * third_term / likelihood

    def expected_number_of_purchases_up_to_time(self, t):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population.

        Parameters:
            t: a scalar or array of times.

        Returns: a scalar or array
        """
        r, alpha, s, beta = self._unload_params('r', 'alpha', 's', 'beta')
        first_term = r * beta / alpha / (s - 1)
        second_term = 1 - (beta / (beta + t)) ** (s - 1)
        return first_term * second_term


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

    def fit(self, frequency, recency, T, iterative_fitting=1, initial_params=None, verbose=False):
        """
        This methods fits the data to the BG/NBD model.

        Parameters:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            iterative_fitting: perform `iterative_fitting` additional fits to find the best
                parameters for the model. Setting to 0 will improve performances but possibly
                hurt estimates.
            initial_params: set the initial parameters for the fitter.
            verbose: set to true to print out convergence diagnostics.


        Returns:
            self, with additional properties and methods like params_ and predict

        """
        frequency = asarray(frequency)
        recency = asarray(recency)
        T = asarray(T)
        _check_inputs(frequency, recency, T)

        self._scale = _scale_time(T)
        scaled_recency = recency * self._scale
        scaled_T = T * self._scale

        params, self._negative_log_likelihood_ = _fit(self._negative_log_likelihood,
                                                      [frequency, scaled_recency, scaled_T, self.penalizer_coef],
                                                      iterative_fitting,
                                                      initial_params,
                                                      4,
                                                      verbose)

        self.params_ = OrderedDict(zip(['r', 'alpha', 'a', 'b'], params))
        self.params_['alpha'] /= self._scale

        self.data = DataFrame(vconcat[frequency, recency, T], columns=['frequency', 'recency', 'T'])
        self.generate_new_data = lambda size=1: beta_geometric_nbd_model(T, *self._unload_params('r', 'alpha', 'a', 'b'), size=size)

        self.predict = self.conditional_expected_number_of_purchases_up_to_time
        return self

    @staticmethod
    def _negative_log_likelihood(params, freq, rec, T, penalizer_coef):
        if npany(asarray(params) <= 0):
            return np.inf

        r, alpha, a, b = params

        A_1 = special.gammaln(r + freq) - special.gammaln(r) + r * log(alpha)
        A_2 = special.gammaln(a + b) + special.gammaln(b + freq) - special.gammaln(b) - special.gammaln(a + b + freq)
        A_3 = -(r + freq) * log(alpha + T)

        d = vconcat[ones_like(freq), (freq > 0)]
        A_4 = log(a) - log(b + freq - 1) - (r + freq) * log(rec + alpha)
        A_4[isnan(A_4) | isinf(A_4)] = 0
        penalizer_term = penalizer_coef * log(params).sum()
        return -(A_1 + A_2 + misc.logsumexp(vconcat[A_3, A_4], axis=1, b=d)).sum() + penalizer_term

    def expected_number_of_purchases_up_to_time(self, t):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population.

        Parameters:
            t: a scalar or array of times.

        Returns: a scalar or array
        """
        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')
        hyp = special.hyp2f1(r, b, a + b - 1, t / (alpha + t))
        return (a + b - 1) / (a - 1) * (1 - hyp * (alpha / (alpha + t)) ** r)

    def conditional_expected_number_of_purchases_up_to_time(self, t, frequency, recency, T):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population, given they have purchase history (frequency, recency, T)

        Parameters:
            t: a scalar or array of times.
            frequency: a scalar: historical frequency of customer.
            recency: a scalar: historical recency of customer.
            T: a scalar: age of the customer.

        Returns: a scalar or array
        """
        x = frequency
        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')

        hyp_term = special.hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
        first_term = (a + b + x - 1) / (a - 1)
        second_term = (1 - hyp_term * ((alpha + T) / (alpha + t + T)) ** (r + x))
        numerator = first_term * second_term

        denominator = 1 + (x > 0) * (a / (b + x - 1)) * ((alpha + T) / (alpha + recency)) ** (r + x)

        return numerator / denominator

    def conditional_probability_alive(self, frequency, recency, T):
        """
        Compute the probability that a customer with history (frequency, recency, T) is currently
        alive. From http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf

        Parameters:
            frequency: a scalar: historical frequency of customer.
            recency: a scalar: historical recency of customer.
            T: a scalar: age of the customer.

        Returns: a scalar

        """
        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')
        return 1. / (1 + (frequency > 0) * (a / (b + frequency - 1)) * ((alpha + T) / (alpha + recency)) ** (r + frequency))

    def conditional_probability_alive_matrix(self, max_frequency=None, max_recency=None):
        """
        Compute the probability alive matrix
        Parameters:
            max_frequency: the maximum frequency to plot. Default is max observed frequency.
            max_recency: the maximum recency to plot. This also determines the age of the customer.
                Default to max observed age.

        Returns a matrix of the form [t_x: historical recency, x: historical frequency]

        """

        max_frequency = max_frequency or int(self.data['frequency'].max())
        max_recency = max_recency or int(self.data['T'].max())

        Z = np.zeros((max_recency + 1, max_frequency + 1))
        for i, t_x in enumerate(np.arange(max_recency + 1)):
            for j, x in enumerate(np.arange(max_frequency + 1)):
                Z[i, j] = self.conditional_probability_alive(x, t_x, max_recency)

        return Z

    def probability_of_n_purchases_up_to_time(self, t, n):
        """
        Compute the probability of

        P( N(t) = n | model )

        where N(t) is the number of repeat purchases a customer makes in t units of time.
        """

        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')

        first_term = special.beta(a, b + n) / special.beta(a, b) * special.gamma(r + n) / special.gamma(r) / special.gamma(n + 1) * (alpha / (alpha + t)) ** r * (t / (alpha + t)) ** n
        if n > 0:
            finite_sum = np.sum([special.gamma(r + j) / special.gamma(r) / special.gamma(j + 1) * (t / (alpha + t)) ** j for j in range(0, n)])
            second_term = special.beta(a + 1, b + n - 1) / special.beta(a, b) * (1 - (alpha / (alpha + t)) ** r * finite_sum)
        else:
            second_term = 0
        return first_term + second_term


class ModifiedBetaGeoFitter(BetaGeoFitter):

    """

    Also known as the MBG/NBD model. Based on [1,2], this model has the following assumptions:
    1) Each individual, i, has a hidden lambda_i and p_i parameter
    2) These come from a population wide Gamma and a Beta distribution respectively.
    3) Individuals purchases follow a Poisson process with rate lambda_i*t .
    4) At the beginning of their lifetime and after each purchase, an individual has a
       p_i probability of dieing (never buying again).

    [1] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
        "Empirical validation and comparison of models for customer base analysis,"
        International Journal of Research in Marketing, 24 (3), 201-209.
    [2] Wagner, U. and Hoppe D. (2008), "Erratum on the MBG/NBD Model," International Journal
        of Research in Marketing, 25 (3), 225-226.


    """

    def __init__(self, penalizer_coef=0.):
        super(self.__class__, self).__init__(penalizer_coef)

    def fit(self, frequency, recency, T, iterative_fitting=1, initial_params=None, verbose=False):
        """
        This methods fits the data to the MBG/NBD model.

        Parameters:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            iterative_fitting: perform `iterative_fitting` additional fits to find the best
                parameters for the model. Setting to 0 will improve performances but possibly
                hurt estimates.
            initial_params: set the initial parameters for the fitter.
            verbose: set to true to print out convergence diagnostics.


        Returns:
            self, with additional properties and methods like params_ and predict

        """
        super(self.__class__, self).fit(frequency, recency, T, iterative_fitting, initial_params, verbose)  # although the partent method is called, this class's _negative_log_likelihood is referenced
        self.generate_new_data = lambda size=1: modified_beta_geometric_nbd_model(T, *self._unload_params('r', 'alpha', 'a', 'b'), size=size)  # this needs to be reassigned from the parent method
        return self

    @staticmethod
    def _negative_log_likelihood(params, freq, rec, T, penalizer_coef):
        if npany(asarray(params) <= 0):
            return np.inf

        r, alpha, a, b = params

        A_1 = special.gammaln(r + freq) - special.gammaln(r) + r * log(alpha)
        A_2 = special.gammaln(a + b) + special.gammaln(b + freq + 1) - special.gammaln(b) - special.gammaln(a + b + freq + 1)
        A_3 = -(r + freq) * log(alpha + T)
        A_4 = log(a) - log(b + freq) + (r + freq) * (log(alpha + T) - log(alpha + rec))

        penalizer_term = penalizer_coef * log(params).sum()
        return -(A_1 + A_2 + A_3 + log(exp(A_4) + 1.)).sum() + penalizer_term

    def expected_number_of_purchases_up_to_time(self, t):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population.

        Parameters:
            t: a scalar or array of times.

        Returns: a scalar or array
        """
        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')
        hyp = special.hyp2f1(r, b + 1, a + b, t / (alpha + t))
        return b / (a - 1) * (1 - hyp * (alpha / (alpha + t)) ** r)

    def conditional_expected_number_of_purchases_up_to_time(self, t, frequency, recency, T):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population, given they have purchase history (frequency, recency, T)
        See Wagner, U. and Hoppe D. (2008).

        Parameters:
            t: a scalar or array of times.
            frequency: a scalar: historical frequency of customer.
            recency: a scalar: historical recency of customer.
            T: a scalar: age of the customer.

        Returns: a scalar or array
        """
        x = frequency
        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')

        hyp_term = special.hyp2f1(r + x, b + x + 1, a + b + x, t / (alpha + T + t))
        first_term = (a + b + x) / (a - 1)
        second_term = (1 - hyp_term * ((alpha + T) / (alpha + t + T)) ** (r + x))
        numerator = first_term * second_term

        denominator = 1 + (a / (b + x)) * ((alpha + T) / (alpha + recency)) ** (r + x)

        return numerator / denominator

    def conditional_probability_alive(self, frequency, recency, T):
        """
        Compute the probability that a customer with history (frequency, recency, T) is currently
        alive. From http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf

        Parameters:
            frequency: a scalar: historical frequency of customer.
            recency: a scalar: historical recency of customer.
            T: a scalar: age of the customer.

        Returns: a scalar

        """
        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')
        return 1. / (1 + (a / (b + frequency)) * ((alpha + T) / (alpha + recency)) ** (r + frequency))

    def conditional_probability_alive_matrix(self, max_frequency=None, max_recency=None):
        """
        Compute the probability alive matrix
        Parameters:
            max_frequency: the maximum frequency to plot. Default is max observed frequency.
            max_recency: the maximum recency to plot. This also determines the age of the customer.
                Default to max observed age.

        Returns a matrix of the form [t_x: historical recency, x: historical frequency]
        """
        return super(self.__class__, self).conditional_probability_alive_matrix(max_frequency, max_recency)

    def probability_of_n_purchases_up_to_time(self, t, n):
        """
        Compute the probability of

        P( N(t) = n | model )

        where N(t) is the number of repeat purchases a customer makes in t units of time.
        """

        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')

        first_term = special.beta(a, b + n + 1) / special.beta(a, b) * special.gamma(r + n) / special.gamma(r) / special.gamma(n + 1) * (alpha / (alpha + t)) ** r * (t / (alpha + t)) ** n
        finite_sum = np.sum([special.gamma(r + j) / special.gamma(r) / special.gamma(j + 1) * (t / (alpha + t)) ** j for j in range(0, n)])
        second_term = special.beta(a + 1, b + n) / special.beta(a, b) * (1 - (alpha / (alpha + t)) ** r * finite_sum)

        return first_term + second_term
