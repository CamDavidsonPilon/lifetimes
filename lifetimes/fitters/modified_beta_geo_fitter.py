# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import warnings

import autograd.numpy as np
from autograd.numpy import log, logaddexp
from autograd.scipy.special import gammaln, beta, gamma
from scipy.special import hyp2f1

from lifetimes import BetaGeoFitter
from lifetimes.generate_data import modified_beta_geometric_nbd_model


class ModifiedBetaGeoFitter(BetaGeoFitter):
    r"""
    Also known as the MBG/NBD model.

    Based on [5]_, [6]_, this model has the following assumptions:
    1) Each individual, ``i``, has a hidden ``lambda_i`` and ``p_i`` parameter
    2) These come from a population wide Gamma and a Beta distribution
       respectively.
    3) Individuals purchases follow a Poisson process with rate :math:`\lambda_i*t` .
    4) At the beginning of their lifetime and after each purchase, an
       individual has a p_i probability of dieing (never buying again).

    References
    ----------
    .. [5] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
       "Empirical validation and comparison of models for customer base
       analysis,"
       International Journal of Research in Marketing, 24 (3), 201-209.
    .. [6] Wagner, U. and Hoppe D. (2008), "Erratum on the MBG/NBD Model,"
       International Journal of Research in Marketing, 25 (3), 225-226.

    Attributes
    -----------
    penalizer_coef: float
        The coefficient applied to an l2 norm on the parameters
    params_: :obj: Series
        The fitted parameters of the model
    data: :obj: DataFrame
        A DataFrame with the values given in the call to `fit`
    variance_matrix_: :obj: DataFrame
        A DataFrame with the variance matrix of the parameters.
    confidence_intervals_: :obj: DataFrame
        A DataFrame 95% confidence intervals of the parameters
    standard_errors_: :obj: Series
        A Series with the standard errors of the parameters
    summary: :obj: DataFrame
        A DataFrame containing information about the fitted parameters
    """

    def __init__(self, penalizer_coef=0.0):
        """Initialization, set penalizer_coef."""
        super(ModifiedBetaGeoFitter, self).__init__(penalizer_coef)

    def fit(
        self, frequency, recency, T, weights=None, initial_params=None, verbose=False, tol=1e-7, index=None, **kwargs
    ):
        """
        Fit the data to the MBG/NBD model.

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
            log-likelihood, the log-likelihood is calculated for each
            pattern and multiplied by the number of customers with
            that pattern.
        verbose : bool, optional
            set to true to print out convergence diagnostics.
        tol : float, optional
            tolerance for termination of the function minimization process.
        index: array_like, optional
            index for resulted DataFrame which is accessible via self.data
        kwargs:
            key word arguments to pass to the scipy.optimize.minimize
            function as options dict

        Returns
        -------
        ModifiedBetaGeoFitter:
            With additional properties and methods like ``params_`` and ``predict``

        """
        # although the parent method is called, this class's
        # _negative_log_likelihood is referenced
        super(ModifiedBetaGeoFitter, self).fit(
            frequency, recency, T, weights, initial_params, verbose, tol, index=index, **kwargs
        )
        # this needs to be reassigned from the parent method
        self.generate_new_data = lambda size=1: modified_beta_geometric_nbd_model(
            T, *self._unload_params("r", "alpha", "a", "b"), size=size
        )

        self.variance_matrix_ = self._compute_variance_matrix()
        self.standard_errors_ = self._compute_standard_errors()
        self.confidence_intervals_ = self._compute_confidence_intervals()
        return self

    @staticmethod
    def _negative_log_likelihood(log_params, freq, rec, T, weights, penalizer_coef):
        warnings.simplefilter(action="ignore", category=FutureWarning)

        params = np.exp(log_params)
        r, alpha, a, b = params

        A_1 = gammaln(r + freq) - gammaln(r) + r * log(alpha)
        A_2 = gammaln(a + b) + gammaln(b + freq + 1) - gammaln(b) - gammaln(a + b + freq + 1)
        A_3 = -(r + freq) * log(alpha + T)
        A_4 = log(a) - log(b + freq) + (r + freq) * (log(alpha + T) - log(alpha + rec))

        penalizer_term = penalizer_coef * sum(params ** 2)
        return -(weights * (A_1 + A_2 + A_3 + logaddexp(A_4, 0))).sum() / weights.sum() + penalizer_term

    def expected_number_of_purchases_up_to_time(self, t):
        """
        Return expected number of repeat purchases up to time t.

        Calculate the expected number of repeat purchases up to time t for a
        randomly choose individual from the population.

        Parameters
        ----------
        t: array_like
            times to calculate the expectation for

        Returns
        -------
        array_like

        """
        r, alpha, a, b = self._unload_params("r", "alpha", "a", "b")
        hyp = hyp2f1(r, b + 1, a + b, t / (alpha + t))
        return b / (a - 1) * (1 - hyp * (alpha / (alpha + t)) ** r)

    def conditional_expected_number_of_purchases_up_to_time(self, t, frequency, recency, T):
        """
        Conditional expected number of repeat purchases up to time t.

        Calculate the expected number of repeat purchases up to time t for a
        randomly choose individual from the population, given they have
        purchase history (frequency, recency, T)
        See Wagner, U. and Hoppe D. (2008).

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
        x = frequency
        r, alpha, a, b = self._unload_params("r", "alpha", "a", "b")

        hyp_term = hyp2f1(r + x, b + x + 1, a + b + x, t / (alpha + T + t))
        first_term = (a + b + x) / (a - 1)
        second_term = 1 - hyp_term * ((alpha + T) / (alpha + t + T)) ** (r + x)
        numerator = first_term * second_term

        denominator = 1 + (a / (b + x)) * ((alpha + T) / (alpha + recency)) ** (r + x)

        return numerator / denominator

    def conditional_probability_alive(self, frequency, recency, T):
        """
        Conditional probability alive.

        Compute the probability that a customer with history (frequency,
        recency, T) is currently alive.
        From https://www.researchgate.net/publication/247219660_Empirical_validation_and_comparison_of_models_for_customer_base_analysis
        Appendix A, eq. (5)

        Parameters
        ----------
        frequency: array or float
            historical frequency of customer.
        recency: array or float
            historical recency of customer.
        T: array or float
            age of the customer.

        Returns
        -------
        array:
            value representing probability of being alive

        """
        r, alpha, a, b = self._unload_params("r", "alpha", "a", "b")
        return np.atleast_1d(1.0 / (1 + (a / (b + frequency)) * ((alpha + T) / (alpha + recency)) ** (r + frequency)))

    def probability_of_n_purchases_up_to_time(self, t, n):
        r"""
        Compute the probability of n purchases up to time t.

        .. math::  P( N(t) = n | \text{model} )

        where N(t) is the number of repeat purchases a customer makes in t
        units of time.

        Parameters
        ----------
        t: float
            number units of time
        n: int
            number of purchases

        Returns
        -------
        float:
            Probability to have n purchases up to t units of time

        """
        r, alpha, a, b = self._unload_params("r", "alpha", "a", "b")
        _j = np.arange(0, n)

        first_term = (
            beta(a, b + n + 1)
            / beta(a, b)
            * gamma(r + n)
            / gamma(r)
            / gamma(n + 1)
            * (alpha / (alpha + t)) ** r
            * (t / (alpha + t)) ** n
        )
        finite_sum = (gamma(r + _j) / gamma(r) / gamma(_j + 1) * (t / (alpha + t)) ** _j).sum()
        second_term = beta(a + 1, b + n) / beta(a, b) * (1 - (alpha / (alpha + t)) ** r * finite_sum)

        return first_term + second_term
