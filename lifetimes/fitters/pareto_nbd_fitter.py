"""Pareto/NBD model."""
from __future__ import print_function
from __future__ import division
from collections import OrderedDict

import numpy as np
from numpy import log, exp, logaddexp, asarray, any as npany, c_ as vconcat
from pandas import DataFrame
from scipy.special import gammaln, hyp2f1, betaln
from scipy import misc

from . import BaseFitter
from ..utils import _fit, _check_inputs, _scale_time
from ..generate_data import pareto_nbd_model


class ParetoNBDFitter(BaseFitter):
    """
    Pareto NBD fitter [7]_.

    Parameters
    ----------
    penalizer_coef: float
        The coefficient applied to an l2 norm on the parameters

    Attributes
    ----------
    penalizer_coef: float
        The coefficient applied to an l2 norm on the parameters
    params_: :obj: OrderedDict
        The fitted parameters of the model
    data: :obj: DataFrame
        A DataFrame with the columns given in the call to `fit`

    References
    ----------
    .. [7] David C. Schmittlein, Donald G. Morrison and Richard Colombo
       Management Science,Vol. 33, No. 1 (Jan., 1987), pp. 1-24
      "Counting Your Customers: Who Are They and What Will They Do Next,"

    """

    def __init__(self, penalizer_coef=0.0):
        """Initialization, set penalizer_coef."""
        self.penalizer_coef = penalizer_coef

    def fit(self, frequency, recency, T, weights=None, iterative_fitting=1,
            initial_params=None, verbose=False, tol=1e-4, index=None,
            fit_method='Nelder-Mead', maxiter=2000, **kwargs):
        """
        Pareto/NBD model fitter.

        Parameters
        ----------
        frequency: array_like
            the frequency vector of customers' purchases
            (denoted x in literature).
        recency: array_like
            the recency vector of customers' purchases
            (denoted t_x in literature).
        T: array_like
            customers' age (time units since first purchase)
        weights: None or array_like
            Number of customers with given frequency/recency/T,
            defaults to 1 if not specified. Fader and
            Hardie condense the individual RFM matrix into all
            observed combinations of frequency/recency/T. This
            parameter represents the count of customers with a given
            purchase pattern. Instead of calculating individual
            loglikelihood, the loglikelihood is calculated for each
            pattern and multiplied by the number of customers with
            that pattern.
        iterative_fitting: int, optional
            perform iterative_fitting fits over random/warm-started initial params
        initial_params: array_like, optional
            set the initial parameters for the fitter.
        verbose : bool, optional
            set to true to print out convergence diagnostics.
        tol : float, optional
            tolerance for termination of the function minimization process.
        index: array_like, optional
            index for resulted DataFrame which is accessible via self.data
        fit_method : string, optional
            fit_method to passing to scipy.optimize.minimize
        maxiter : int, optional
            max iterations for optimizer in scipy.optimize.minimize will be
            overwritten if setted in kwargs.
        kwargs:
            key word arguments to pass to the scipy.optimize.minimize
            function as options dict

        Returns
        -------
        ParetoNBDFitter
            with additional properties like params_ and methods like predict

        """
        frequency = asarray(frequency).astype(int)
        recency = asarray(recency)
        T = asarray(T)

        if weights is None:
            weights = np.ones(recency.shape[0], dtype=np.int64)
        else:
            weights = asarray(weights)


        _check_inputs(frequency, recency, T)

        self._scale = _scale_time(T)
        scaled_recency = recency * self._scale
        scaled_T = T * self._scale

        params, self._negative_log_likelihood_ = _fit(
            self._negative_log_likelihood,
            [frequency, scaled_recency, scaled_T, weights, self.penalizer_coef],
            iterative_fitting,
            initial_params,
            4,
            verbose,
            tol,
            fit_method,
            maxiter,
            **kwargs)

        self.params_ = OrderedDict(zip(['r', 'alpha', 's', 'beta'], params))
        self.params_['alpha'] /= self._scale
        self.params_['beta'] /= self._scale

        self.data = DataFrame(vconcat[frequency, recency, T],
                              columns=['frequency', 'recency', 'T'])
        if index is not None:
            self.data.index = index
        self.generate_new_data = lambda size=1: pareto_nbd_model(T, *params,
                                                                 size=size)

        self.predict = self.conditional_expected_number_of_purchases_up_to_time
        return self

    @staticmethod
    def _log_A_0(params, freq, recency, age):
        """log_A_0."""
        r, alpha, s, beta = params

        if alpha < beta:
            min_of_alpha_beta, max_of_alpha_beta, t = (alpha, beta, r + freq)
        else:
            min_of_alpha_beta, max_of_alpha_beta, t = (beta, alpha, s + 1)
        abs_alpha_beta = max_of_alpha_beta - min_of_alpha_beta

        rsf = r + s + freq
        p_1 = hyp2f1(rsf, t, rsf + 1., abs_alpha_beta /
                     (max_of_alpha_beta + recency))
        q_1 = max_of_alpha_beta + recency
        p_2 = hyp2f1(rsf, t, rsf + 1., abs_alpha_beta /
                     (max_of_alpha_beta + age))
        q_2 = max_of_alpha_beta + age

        try:
            size = len(freq)
            sign = np.ones(size)
        except TypeError:
            sign = 1

        return (misc.logsumexp([log(p_1) + rsf * log(q_2), log(p_2) +
                rsf * log(q_1)], axis=0, b=[sign, -sign]) -
                rsf * log(q_1 * q_2))

    @staticmethod
    def _conditional_log_likelihood(params, freq, rec, T):

        r, alpha, s, beta = params
        x = freq

        r_s_x = r + s + x

        A_1 = gammaln(r + x) - gammaln(r) + r * log(alpha) + s * log(beta)
        log_A_0 = ParetoNBDFitter._log_A_0(params, x, rec, T)

        A_2 = logaddexp(-(r + x) * log(alpha + T) - s * log(beta + T),
                        log(s) + log_A_0 - log(r_s_x))

        return A_1 + A_2

    @staticmethod
    def _negative_log_likelihood(params, freq, rec, T, weights, penalizer_coef):

        if npany(asarray(params) <= 0.):
            return np.inf

        conditional_log_likelihood = ParetoNBDFitter._conditional_log_likelihood(params, freq, rec, T)
        penalizer_term = penalizer_coef * sum(np.asarray(params) ** 2)

        return -(weights * conditional_log_likelihood).mean() + penalizer_term

    def conditional_expected_number_of_purchases_up_to_time(self, t, frequency,
                                                            recency, T):
        """
        Conditional expected number of purchases up to time.

        Calculate the expected number of repeat purchases up to time t for a
        randomly choose individual from the population, given they have
        purchase history (frequency, recency, T)

        Parameters
        ----------
        t: array_like
            times to calculate the expectation for.
        frequency: array_like
            historical frequency of customer.
        recency: array_like
            historical recency of customer.
        T: array_like
            age of the customer.

        Returns
        -------
        array_like

        """
        x, t_x = frequency, recency
        params = self._unload_params('r', 'alpha', 's', 'beta')
        r, alpha, s, beta = params

        likelihood = self._conditional_log_likelihood(params, x, t_x, T)
        first_term = gammaln(r + x) - gammaln(r) + r * log(alpha) + s * \
            log(beta) - (r + x) * log(alpha + T) - s * log(beta + T)
        second_term = log(r + x) + log(beta + T) - log(alpha + T)
        third_term = log((1 - ((beta + T) / (beta + T + t)) ** (s - 1)) /
                         (s - 1))
        return exp(first_term + second_term + third_term - likelihood)

    def conditional_probability_alive(self, frequency, recency, T):
        """
        Conditional probability alive.

        Compute the probability that a customer with history
        (frequency, recency, T) is currently alive.
        From paper:
        http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf

        Parameters
        ----------
        frequency: float
            historical frequency of customer.
        recency: float
            historical recency of customer.
        T: float
            age of the customer.

        Returns
        -------
        float
            value representing a probability

        """
        x, t_x = frequency, recency
        r, alpha, s, beta = self._unload_params('r', 'alpha', 's', 'beta')
        A_0 = self._log_A_0([r, alpha, s, beta], x, t_x, T)
        return 1. / (1. + exp(log(s) - log(r + s + x) +
                              (r + x) * log(alpha + T) + s * log(beta + T) + A_0))

    def conditional_probability_alive_matrix(self, max_frequency=None,
                                             max_recency=None):
        """
        Compute the probability alive matrix.

        Parameters
        ----------
        max_frequency: float, optional
            the maximum frequency to plot. Default is max observed frequency.
        max_recency: float, optional
            the maximum recency to plot. This also determines the age of the
            customer. Default to max observed age.

        Returns
        -------
        matrix:
            A matrix of the form [t_x: historical recency, x: historical frequency]

        """
        max_frequency = max_frequency or int(self.data['frequency'].max())
        max_recency = max_recency or int(self.data['T'].max())

        Z = np.zeros((max_recency + 1, max_frequency + 1))
        for i, recency in enumerate(np.arange(max_recency + 1)):
            for j, frequency in enumerate(np.arange(max_frequency + 1)):
                Z[i, j] = self.conditional_probability_alive(frequency,
                                                             recency,
                                                             max_recency)

        return Z

    def expected_number_of_purchases_up_to_time(self, t):
        """
        Return expected number of repeat purchases up to time t.

        Calculate the expected number of repeat purchases up to time t for a
        randomly choose individual from the population.

        Parameters
        ----------
        t: array_like
            times to calculate the expectation for.

        Returns
        -------
        array_like

        """
        r, alpha, s, beta = self._unload_params('r', 'alpha', 's', 'beta')
        first_term = r * beta / alpha / (s - 1)
        second_term = 1 - (beta / (beta + t)) ** (s - 1)
        return first_term * second_term

    def conditional_probability_of_n_purchases_up_to_time(self, n, t, frequency, recency, T):
        """
        Return conditional probability of n purchases up to time t.

        Calculate the probability of n purchases up to time t for an individual
        with history frequency, recency and T (age).

        From paper:
        http://www.brucehardie.com/notes/028/pareto_nbd_conditional_pmf.pdf

        Parameters
        ----------
        n: int
            number of purchases.
        t: a scalar
            time up to which probability should be calculated.
        frequency: float
            historical frequency of customer.
        recency: float
            historical recency of customer.
        T: float
            age of the customer.

        Returns
        -------
        array_like

        """
        if t <= 0:
            return 0

        x, t_x = frequency, recency
        params = self._unload_params('r', 'alpha', 's', 'beta')
        r, alpha, s, beta = params

        if alpha < beta:
            min_of_alpha_beta, max_of_alpha_beta, p, p_l_1, p_l_2 = (alpha, beta, r + x + n, r + x, r + x + 1)
        else:
            min_of_alpha_beta, max_of_alpha_beta, p, p_l_1, p_l_2 = (beta, alpha, s + 1, s + 1, s)
        abs_alpha_beta = max_of_alpha_beta - min_of_alpha_beta

        log_l = self._conditional_log_likelihood(params, x, t_x, T)
        log_p_zero = gammaln(r + x) + r * log(alpha) + s * log(beta) - (
            gammaln(r) + (r + x) * log(alpha + T) + s * log(beta + T) +
            log_l
        )
        log_B_one = gammaln(r + x + n) + r * log(alpha) + s * log(beta) - (
            gammaln(r) + (r + x + n) * log(alpha + T + t) + s * log(beta + T + t)
        )
        log_B_two = r * log(alpha) + s * log(beta) + gammaln(r + s + x) + betaln(r + x + n, s + 1) + \
            log(hyp2f1(r + s + x, p, r + s + x + n + 1, abs_alpha_beta / (max_of_alpha_beta + T))) - (
                gammaln(r) + gammaln(s) + (r + s + x) * log(max_of_alpha_beta + T)
        )

        def _log_B_three(i):
            return r * log(alpha) + s * log(beta) + gammaln(r + s + x + i) + betaln(r + x + n, s + 1) + \
                log(hyp2f1(r + s + x + i, p, r + s + x + n + 1, abs_alpha_beta / (max_of_alpha_beta + T + t))) - (
                    gammaln(r) + gammaln(s) + (r + s + x + i) * log(max_of_alpha_beta + T + t)
            )

        zeroth_term = (n == 0) * (1 - exp(log_p_zero))
        first_term = n * log(t) - gammaln(n + 1) + log_B_one - log_l
        second_term = log_B_two - log_l
        third_term = misc.logsumexp(
            [i * log(t) - gammaln(i + 1) + _log_B_three(i) - log_l for i in range(n + 1)],
            axis=0
        )

        try:
            size = len(x)
            sign = np.ones(size)
        except TypeError:
            sign = 1

        # In some scenarios (e.g. large n) tiny numerical errors in the calculation of second_term and third_term
        # cause sumexp to be ever so slightly negative and logsumexp throws an error. Hence we ignore the sign here.
        return zeroth_term + exp(misc.logsumexp(
            [first_term, second_term, third_term], b=[sign, sign, -sign],
            axis=0,
            return_sign=True
        )[0])
