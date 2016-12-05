from __future__ import print_function
from collections import OrderedDict

import numpy as np
import pandas as pd
from numpy import log, exp, logaddexp, asarray, any as npany, c_ as vconcat,\
    isinf, isnan, ones_like
from pandas import DataFrame

from scipy.special import gammaln, hyp2f1, beta, gamma, betaln, binom
from scipy import misc

from lifetimes.utils import _fit, _scale_time, _check_inputs, customer_lifetime_value
from lifetimes.generate_data import pareto_nbd_model, beta_geometric_nbd_model, modified_beta_geometric_nbd_model

__all__ = ['BetaGeoFitter', 'ParetoNBDFitter', 'GammaGammaFitter', 'ModifiedBetaGeoFitter', 'BetaGeoBetaBinomFitter']


class BaseFitter(object):

    def __repr__(self):
        classname = self.__class__.__name__
        try:
            param_str = ", ".join("%s: %.2f" % (param, value) for param, value in sorted(self.params_.items()))
            return "<lifetimes.%s: fitted with %d subjects, %s>" % (classname, self.data.shape[0], param_str)
        except AttributeError:
            return "<lifetimes.%s>" % classname

    def _unload_params(self, *args):
        if not hasattr(self, 'params_'):
            raise ValueError("Model has not been fit yet. Please call the .fit method first.")
        return [self.params_[x] for x in args]


class BetaGeoBetaBinomFitter(BaseFitter):

    """
    Also known as the Beta-Geometric/Beta-Binomial Model Model [1].

    Future purchases opportunities are treated as discrete points in time. In the literature,
    the model provides a better fit than the Pareto/NBD model for a nonprofit organization
    with regular giving patterns.

    The model is estimated with a recency-frequency matrix with n transaction opportunities.

    [1] Fader, Peter S., Bruce G.S. Hardie, and Jen Shang (2010),
        "Customer-Base Analysis in a Discrete-Time Noncontractual Setting,"
        Marketing Science, 29 (6), 1086-1108.

    """

    def __init__(self, penalizer_coef=0.):
        self.penalizer_coef = penalizer_coef

    @staticmethod
    def _loglikelihood(params, x, tx, T):

        alpha, beta, gamma, delta = params

        beta_ab = betaln(alpha, beta)
        beta_gd = betaln(gamma, delta)

        indiv_loglike = (betaln(alpha + x, beta + T - x) - beta_ab +
                         betaln(gamma, delta + T) - beta_gd)

        recency_T = T - tx - 1

        J = np.arange(recency_T.max() + 1)

        @np.vectorize
        def _sum(x, tx, recency_T):
            j = J[:recency_T + 1]
            return log(
                np.sum(exp(betaln(alpha + x, beta + tx - x + j) - beta_ab +
                           betaln(gamma + 1, delta + tx + j) - beta_gd)))

        s = _sum(x, tx, recency_T)
        indiv_loglike = logaddexp(indiv_loglike, s)

        return indiv_loglike

    @staticmethod
    def _negative_log_likelihood(params, frequency, recency, n, n_custs, penalizer_coef=0):
        penalizer_term = penalizer_coef * sum(np.asarray(params) ** 2)
        return -np.mean(BetaGeoBetaBinomFitter._loglikelihood(params, frequency, recency, n) * n_custs) + penalizer_term

    def fit(self, frequency, recency, n, n_custs, verbose=False, tol=1e-4, iterative_fitting=1):
        """
        Fit the BG/BB model.

        Parameters:
            frequency: Total periods with observed transactions
            recency: Period of most recent transaction
            n: Number of transaction opportunities
            n_custs: Number of customers with given frequency/recency/T. Fader and Hardie condense
                    the individual RFM matrix into all observed combinations of frequency/recency/T. This
                    parameter represents the count of customers with a given purchase pattern. Instead of
                    calculating individual loglikelihood, the loglikelihood is calculated for each pattern and
                    multiplied by the number of customers with that pattern.

        Returns: self

        """

        frequency = asarray(frequency)
        recency = asarray(recency)
        n = asarray(n)
        n_custs = asarray(n_custs)
        _check_inputs(frequency, recency, n)

        params, self._negative_log_likelihood_ = _fit(self._negative_log_likelihood,
                                                      [frequency, recency, n, n_custs, self.penalizer_coef],
                                                      iterative_fitting,
                                                      np.ones(4),
                                                      4,
                                                      verbose,
                                                      tol)
        self.params_ = OrderedDict(zip(['alpha', 'beta', 'gamma', 'delta'], params))
        self.data = DataFrame(vconcat[frequency, recency, n, n_custs],
                              columns=['frequency', 'recency', 'n', 'n_custs'])

        return self

    def conditional_expected_number_of_purchases_up_to_time(self, t):
        """
        Conditional expected purchases in future time period.

        The  expected  number  of  future  transactions across the next t transaction
        opportunities by a customer with purchase history (x, tx, n).

        E(X(n, n+n*)|alpha, beta, gamma, delta, frequency, recency, n)

        See (13) in Fader & Hardie 2010.

        Parameters:
            t: scalar or array of time periods (n+t)

        Returns: scalar or array of predicted transactions

        """

        x = self.data['frequency']
        tx = self.data['recency']
        n = self.data['n']

        params = self._unload_params('alpha', 'beta', 'gamma', 'delta')
        alpha, beta, gamma, delta = params

        p1 = 1 / exp(BetaGeoBetaBinomFitter._loglikelihood(params, x, tx, n))
        p2 = exp(betaln(alpha + x + 1, beta + n - x) - betaln(alpha, beta))
        p3 = delta / (gamma - 1) * exp(gammaln(gamma + delta) - gammaln(1 + delta))
        p4 = exp(gammaln(1 + delta + n) - gammaln(gamma + delta + n))
        p5 = exp(gammaln(1 + delta + n + t) - gammaln(gamma + delta + n + t))

        return p1 * p2 * p3 * (p4 - p5)

    def conditional_probability_alive(self, m):
        """
        Conditional probability customer is alive at transaction opportunity n + m.

        P(alive at n + m|alpha, beta, gamma, delta, frequency, recency, n)

        See (A10) in Fader and Hardie 2010.

        Parameters:
            m: scalar or array of transaction opportunities

        Returns: scalar or array of alive probabilities

        """

        params = self._unload_params('alpha', 'beta', 'gamma', 'delta')
        alpha, beta, gamma, delta = params

        x = self.data['frequency']
        tx = self.data['recency']
        n = self.data['n']

        p1 = betaln(alpha + x, beta + n - x) - betaln(alpha, beta)
        p2 = betaln(gamma, delta + n + m) - betaln(gamma, delta)
        p3 = BetaGeoBetaBinomFitter._loglikelihood(params, x, tx, n)

        return exp(p1 + p2) / exp(p3)

    def expected_number_of_transactions_in_first_n_periods(self, n):
        """
        Expected number of transactions occurring across first n transaction opportunities. Used by Fader
        and Hardie to assess in-sample fit.

        Pr(X(n) = x|alpha, beta, gamma, delta)

        See (7) in Fader & Hardie 2010.

        Parameters:
            n: scalar, number of transaction opportunities

        Returns: DataFrame of predicted values, indexed by x

        """

        params = self._unload_params('alpha', 'beta', 'gamma', 'delta')
        alpha, beta, gamma, delta = params

        x_counts = self.data.groupby('frequency')['n_custs'].sum()
        x = asarray(x_counts.index)

        p1 = binom(n, x) * exp(betaln(alpha + x, beta + n - x) - betaln(alpha, beta) +
                               betaln(gamma, delta + n) - betaln(gamma, delta))

        I = np.arange(x.min(), n)

        @np.vectorize
        def p2(j, x):
            i = I[j:]
            return np.sum(
                binom(i, x) *
                exp(
                    betaln(alpha + x, beta + i - x) -
                    betaln(alpha, beta) +
                    betaln(gamma + 1, delta + i) -
                    betaln(gamma, delta)
                )
            )

        p1 += np.fromfunction(p2, (x.shape[0],), x=x)

        idx = pd.Index(x, name='frequency')
        return DataFrame(p1 * x_counts.sum(), index=idx, columns=['model'])


class GammaGammaFitter(BaseFitter):

    """
    Fitter for the gamma-gamma model, which is used to estimate the average monetary value of customer transactions.

    This implementation is based on the Excel spreadsheet found in [1]. More details on the derivation and evaluation
    can be found in [2].

    [1] http://www.brucehardie.com/notes/025/
    [2] Peter S. Fader, Bruce G. S. Hardie, and Ka Lok Lee (2005), "RFM and CLV: Using iso-value curves for customer
        base analysis", Journal of Marketing Research, 42 (November), 415-430
    """

    def __init__(self, penalizer_coef=0.0):
        self.penalizer_coef = penalizer_coef

    @staticmethod
    def _negative_log_likelihood(params, frequency, avg_monetary_value, penalizer_coef=0):
        if any(i < 0 for i in params):
            return np.inf

        p, q, v = params

        x = frequency
        m = avg_monetary_value

        negative_log_likelihood_values = (gammaln(p * x + q) -
                                          gammaln(p * x) -
                                          gammaln(q) +
                                          q * np.log(v) +
                                          (p * x - 1) * np.log(m) +
                                          (p * x) * np.log(x) -
                                          (p * x + q) * np.log(x * m + v))
        penalizer_term = penalizer_coef * sum(np.asarray(params) ** 2)
        return -np.mean(negative_log_likelihood_values) + penalizer_term

    def conditional_expected_average_profit(self, frequency=None, monetary_value=None):
        """
        This method computes the conditional expectation of the average profit per transaction
        for a group of one or more customers.

        Parameters:
            frequency: a vector containing the customers' frequencies. Defaults to the whole set of
                frequencies used for fitting the model.
            monetary_value: a vector containing the customers' monetary values. Defaults to the whole set of
                monetary values used for fitting the model.

        Returns:
            the conditional expectation of the average profit per transaction
        """
        if monetary_value is None:
            monetary_value = self.data['monetary_value']
        if frequency is None:
            frequency = self.data['frequency']
        p, q, v = self._unload_params('p', 'q', 'v')
        # The expected average profit is a weighted average of individual monetary value and the population mean.
        individual_weight = p * frequency / (p * frequency + q - 1)
        population_mean = v * p / (q - 1)
        return (1 - individual_weight) * population_mean + individual_weight * monetary_value

    def fit(self, frequency, monetary_value, iterative_fitting=4, initial_params=None, verbose=False, tol=1e-4):
        """
        This methods fits the data to the Gamma/Gamma model.

        Parameters:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            monetary_value: the monetary value vector of customer's purchases (denoted m in literature).
            iterative_fitting: perform iterative_fitting fits over random/warm-started initial params.
            initial_params: set initial params for the iterative fitter.
            verbose: set to true to print out convergence diagnostics.
            tol: tolerance for termination of the function minimization process.

        Returns:
            self, fitted and with parameters estimated
        """
        _check_inputs(frequency, monetary_value=monetary_value)

        params, self._negative_log_likelihood_ = _fit(self._negative_log_likelihood,
                                                      [frequency, monetary_value, self.penalizer_coef],
                                                      iterative_fitting,
                                                      initial_params,
                                                      3,
                                                      verbose,
                                                      tol)

        self.data = DataFrame(vconcat[frequency, monetary_value], columns=['frequency', 'monetary_value'])
        self.params_ = OrderedDict(zip(['p', 'q', 'v'], params))

        return self

    def customer_lifetime_value(self, transaction_prediction_model, frequency, recency, T, monetary_value, time=12, discount_rate=0.01):
        """
        This method computes the average lifetime value for a group of one or more customers.

        Parameters:
            transaction_prediction_model: the model to predict future transactions, literature uses
                pareto/ndb but we can also use a different model like bg
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            monetary_value: the monetary value vector of customer's purchases (denoted m in literature).
            time: the lifetime expected for the user in months. Default: 12
            discount_rate: the monthly adjusted discount rate. Default: 0.01

        Returns:
            Series object with customer ids as index and the estimated customer lifetime values as values
        """
        # use the Gamma-Gamma estimates for the monetary_values
        adjusted_monetary_value = self.conditional_expected_average_profit(frequency, monetary_value)
        return customer_lifetime_value(transaction_prediction_model, frequency, recency, T, adjusted_monetary_value, time, discount_rate)


class ParetoNBDFitter(BaseFitter):

    def __init__(self, penalizer_coef=0.0):
        self.penalizer_coef = penalizer_coef

    def fit(self, frequency, recency, T, iterative_fitting=1, initial_params=None, verbose=False, tol=1e-4):
        """
        This methods fits the data to the Pareto/NBD model.

        Parameters:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            iterative_fitting: perform iterative_fitting fits over random/warm-started initial params
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
                                                      verbose,
                                                      tol)

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
        p_1, q_1 = hyp2f1(rsf, t, rsf + 1., abs_alpha_beta / (max_of_alpha_beta + recency)), (max_of_alpha_beta + recency)
        p_2, q_2 = hyp2f1(rsf, t, rsf + 1., abs_alpha_beta / (max_of_alpha_beta + age)), (max_of_alpha_beta + age)

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

        A_1 = gammaln(r + x) - gammaln(r) + r * log(alpha) + s * log(beta)
        log_A_0 = ParetoNBDFitter._log_A_0(params, freq, rec, T)

        A_2 = logaddexp(-(r + x) * log(alpha + T) - s * log(beta + T), log(s) + log_A_0 - log(r_s_x))

        penalizer_term = penalizer_coef * sum(np.asarray(params) ** 2)
        return -(A_1 + A_2).mean() + penalizer_term

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
        first_term = (gamma(r + x) / gamma(r)) * (alpha ** r * beta ** s) / (alpha + T) ** (r + x) / (beta + T) ** s
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

    def __init__(self, penalizer_coef=0.0):
        self.penalizer_coef = penalizer_coef

    def fit(self, frequency, recency, T, iterative_fitting=1, initial_params=None, verbose=False, tol=1e-4):
        """
        This methods fits the data to the BG/NBD model.

        Parameters:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            iterative_fitting: perform iterative_fitting fits over random/warm-started initial params
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
                                                      verbose,
                                                      tol)

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

        A_1 = gammaln(r + freq) - gammaln(r) + r * log(alpha)
        A_2 = gammaln(a + b) + gammaln(b + freq) - gammaln(b) - gammaln(a + b + freq)
        A_3 = -(r + freq) * log(alpha + T)

        d = vconcat[ones_like(freq), (freq > 0)]
        A_4 = log(a) - log(b + freq - 1) - (r + freq) * log(rec + alpha)
        A_4[isnan(A_4) | isinf(A_4)] = 0
        penalizer_term = penalizer_coef * sum(np.asarray(params) ** 2)
        return -(A_1 + A_2 + misc.logsumexp(vconcat[A_3, A_4], axis=1, b=d)).mean() + penalizer_term

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

        hyp_term = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
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

        return np.fromfunction(self.conditional_probability_alive,
                               (max_frequency + 1, max_recency + 1),
                               T=max_recency).T

    def probability_of_n_purchases_up_to_time(self, t, n):
        """
        Compute the probability of

        P( N(t) = n | model )

        where N(t) is the number of repeat purchases a customer makes in t units of time.
        """

        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')

        first_term = beta(a, b + n) / beta(a, b) * gamma(r + n) / gamma(r) / gamma(n + 1) * (alpha / (alpha + t)) ** r * (t / (alpha + t)) ** n
        if n > 0:
            j = np.arange(0, n)
            finite_sum = (gamma(r + j) / gamma(r) / gamma(j + 1) * (t / (alpha + t)) ** j).sum()
            second_term = beta(a + 1, b + n - 1) / beta(a, b) * (1 - (alpha / (alpha + t)) ** r * finite_sum)
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

    def __init__(self, penalizer_coef=0.0):
        super(self.__class__, self).__init__(penalizer_coef)

    def fit(self, frequency, recency, T, iterative_fitting=1, initial_params=None, verbose=False, tol=1e-4):
        """
        This methods fits the data to the MBG/NBD model.

        Parameters:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
            iterative_fitting: perform iterative_fitting fits over random/warm-started initial params
            initial_params: set the initial parameters for the fitter.
            verbose: set to true to print out convergence diagnostics.


        Returns:
            self, with additional properties and methods like params_ and predict

        """
        super(self.__class__, self).fit(frequency, recency, T, iterative_fitting, initial_params, verbose, tol)  # although the parent method is called, this class's _negative_log_likelihood is referenced
        self.generate_new_data = lambda size=1: modified_beta_geometric_nbd_model(T, *self._unload_params('r', 'alpha', 'a', 'b'), size=size)  # this needs to be reassigned from the parent method
        return self

    @staticmethod
    def _negative_log_likelihood(params, freq, rec, T, penalizer_coef):
        if npany(asarray(params) <= 0):
            return np.inf

        r, alpha, a, b = params

        A_1 = gammaln(r + freq) - gammaln(r) + r * log(alpha)
        A_2 = gammaln(a + b) + gammaln(b + freq + 1) - gammaln(b) - gammaln(a + b + freq + 1)
        A_3 = -(r + freq) * log(alpha + T)
        A_4 = log(a) - log(b + freq) + (r + freq) * (log(alpha + T) - log(alpha + rec))

        penalizer_term = penalizer_coef * sum(np.asarray(params) ** 2)
        return -(A_1 + A_2 + A_3 + logaddexp(A_4, 0)).mean() + penalizer_term

    def expected_number_of_purchases_up_to_time(self, t):
        """
        Calculate the expected number of repeat purchases up to time t for a randomly choose individual from
        the population.

        Parameters:
            t: a scalar or array of times.

        Returns: a scalar or array
        """
        r, alpha, a, b = self._unload_params('r', 'alpha', 'a', 'b')
        hyp = hyp2f1(r, b + 1, a + b, t / (alpha + t))
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

        hyp_term = hyp2f1(r + x, b + x + 1, a + b + x, t / (alpha + T + t))
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
        _j = np.arange(0, n)

        first_term = beta(a, b + n + 1) / beta(a, b) * gamma(r + n) / gamma(r) / gamma(n + 1) * (alpha / (alpha + t)) ** r * (t / (alpha + t)) ** n
        finite_sum = (gamma(r + _j) / gamma(r) / gamma(_j + 1) * (t / (alpha + t)) ** _j).sum()
        second_term = beta(a + 1, b + n) / beta(a, b) * (1 - (alpha / (alpha + t)) ** r * finite_sum)

        return first_term + second_term
