# -*- coding: utf-8 -*-
"""Beta Geo Fitter, also known as BG/NBD model."""

from __future__ import print_function
from __future__ import division
import warnings

import pandas as pd
import autograd.numpy as np
from autograd.scipy.special import gammaln, beta, gamma
from scipy.special import hyp2f1
from scipy.special import expit
from . import BaseFitter
from ..utils import _scale_time, _check_inputs
from ..generate_data import beta_geometric_nbd_model


class BetaGeoFitter(BaseFitter):
    """
    Also known as the BG/NBD model.

    Based on [2]_, this model has the following assumptions:

    1) Each individual, i, has a hidden lambda_i and p_i parameter
    2) These come from a population wide Gamma and a Beta distribution
       respectively.
    3) Individuals purchases follow a Poisson process with rate lambda_i*t .
    4) After each purchase, an individual has a p_i probability of dieing
       (never buying again).

    Parameters
    ----------
    penalizer_coef: float
        The coefficient applied to an l2 norm on the parameters

    Attributes
    ----------
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

    References
    ----------
    .. [2] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
       "Counting Your Customers the Easy Way: An Alternative to the
       Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
    """

    def __init__(
        self, 
        penalizer_coef=0.0
    ):
        """
        Initialization, set penalizer_coef.
        """

        self.penalizer_coef = penalizer_coef

    def fit(
        self, 
        frequency, 
        recency, 
        T, 
        weights=None, 
        initial_params=None, 
        verbose=False, 
        tol=1e-7, 
        index=None, 
        **kwargs
    ):
        """
        Fit a dataset to the BG/NBD model.

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
        initial_params: array_like, optional
            set the initial parameters for the fitter.
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
        BetaGeoFitter
            with additional properties like ``params_`` and methods like ``predict``
        """

        frequency = np.asarray(frequency).astype(int)
        recency = np.asarray(recency)
        T = np.asarray(T)
        _check_inputs(frequency, recency, T)

        if weights is None:
            weights = np.ones_like(recency, dtype=int)
        else:
            weights = np.asarray(weights)

        self._scale = _scale_time(T)
        scaled_recency = recency * self._scale
        scaled_T = T * self._scale

        log_params_, self._negative_log_likelihood_, self._hessian_ = self._fit(
            (frequency, scaled_recency, scaled_T, weights, self.penalizer_coef),
            initial_params,
            4,
            verbose,
            tol,
            **kwargs
        )

        self.params_ = pd.Series(np.exp(log_params_), index=["r", "alpha", "a", "b"])
        self.params_["alpha"] /= self._scale

        self.data = pd.DataFrame({"frequency": frequency, "recency": recency, "T": T, "weights": weights}, index=index)

        self.generate_new_data = lambda size=1: beta_geometric_nbd_model(
            T, *self._unload_params("r", "alpha", "a", "b"), size=size
        )

        self.predict = self.conditional_expected_number_of_purchases_up_to_time

        self.variance_matrix_ = self._compute_variance_matrix()
        self.standard_errors_ = self._compute_standard_errors()
        self.confidence_intervals_ = self._compute_confidence_intervals()

        return self

    @staticmethod
    def _negative_log_likelihood(
        log_params, 
        freq, 
        rec, 
        T, 
        weights, 
        penalizer_coef
    ):
        """
        The following method for calculatating the *log-likelihood* uses the method
        specified in section 7 of [2]_. More information can also be found in [3]_.

        References
        ----------
        .. [2] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
        "Counting Your Customers the Easy Way: An Alternative to the
        Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
        .. [3] http://brucehardie.com/notes/004/
        """

        warnings.simplefilter(action="ignore", category=FutureWarning)

        params = np.exp(log_params)
        r, alpha, a, b = params

        A_1 = gammaln(r + freq) - gammaln(r) + r * np.log(alpha)
        A_2 = gammaln(a + b) + gammaln(b + freq) - gammaln(b) - gammaln(a + b + freq)
        A_3 = -(r + freq) * np.log(alpha + T)
        A_4 = np.log(a) - np.log(b + np.maximum(freq, 1) - 1) - (r + freq) * np.log(rec + alpha)

        max_A_3_A_4 = np.maximum(A_3, A_4)

        penalizer_term = penalizer_coef * sum(params ** 2)
        ll = weights * (A_1 + A_2 + np.log(np.exp(A_3 - max_A_3_A_4) + np.exp(A_4 - max_A_3_A_4) * (freq > 0)) + max_A_3_A_4)

        return -ll.sum() / weights.sum() + penalizer_term

    def conditional_expected_number_of_purchases_up_to_time(
        self, 
        t, 
        frequency, 
        recency, 
        T
    ):
        """
        Conditional expected number of purchases up to time.

        Calculate the expected number of repeat purchases up to time t for a
        randomly chosen individual from the population, given they have
        purchase history (frequency, recency, T).

        This function uses equation (10) from [2]_.

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

        References
        ----------
        .. [2] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
        "Counting Your Customers the Easy Way: An Alternative to the
        Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
        """

        x = frequency
        r, alpha, a, b = self._unload_params("r", "alpha", "a", "b")

        _a = r + x
        _b = b + x
        _c = a + b + x - 1
        _z = t / (alpha + T + t)
        ln_hyp_term = np.log(hyp2f1(_a, _b, _c, _z))

        # if the value is inf, we are using a different but equivalent
        # formula to compute the function evaluation.
        ln_hyp_term_alt = np.log(hyp2f1(_c - _a, _c - _b, _c, _z)) + (_c - _a - _b) * np.log(1 - _z)
        ln_hyp_term = np.where(np.isinf(ln_hyp_term), ln_hyp_term_alt, ln_hyp_term)
        first_term = (a + b + x - 1) / (a - 1)
        second_term = 1 - np.exp(ln_hyp_term + (r + x) * np.log((alpha + T) / (alpha + t + T)))

        numerator = first_term * second_term
        denominator = 1 + (x > 0) * (a / (b + x - 1)) * ((alpha + T) / (alpha + recency)) ** (r + x)

        return numerator / denominator

    def conditional_probability_alive(
        self, 
        frequency, 
        recency, 
        T
    ):
        """
        Compute conditional probability alive.

        Compute the probability that a customer with history
        (frequency, recency, T) is currently alive.

        From http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf

        Parameters
        ----------
        frequency: array or scalar
            historical frequency of customer.
        recency: array or scalar
            historical recency of customer.
        T: array or scalar
            age of the customer.

        Returns
        -------
        array
            value representing a probability
        """

        r, alpha, a, b = self._unload_params("r", "alpha", "a", "b")

        log_div = (r + frequency) * np.log((alpha + T) / (alpha + recency)) + np.log(
            a / (b + np.maximum(frequency, 1) - 1)
        )

        return np.atleast_1d(np.where(frequency == 0, 1.0, expit(-log_div)))

    def conditional_probability_alive_matrix(
        self, 
        max_frequency=None, 
        max_recency=None
    ):
        """
        Compute the probability alive matrix.

        Uses the ``conditional_probability_alive()`` method to get calculate the matrix.

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

        max_frequency = max_frequency or int(self.data["frequency"].max())
        max_recency = max_recency or int(self.data["T"].max())

        return np.fromfunction(
            self.conditional_probability_alive, (max_frequency + 1, max_recency + 1), T=max_recency
        ).T

    def expected_number_of_purchases_up_to_time(
        self, 
        t
    ):
        """
        Calculate the expected number of repeat purchases up to time t.

        Calculate repeat purchases for a randomly chosen individual from the
        population.

        Equivalent to equation (9) of [2]_.

        Parameters
        ----------
        t: array_like
            times to calculate the expection for

        Returns
        -------
        array_like

        References
        ----------
        .. [2] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
        "Counting Your Customers the Easy Way: An Alternative to the
        Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
        """

        r, alpha, a, b = self._unload_params("r", "alpha", "a", "b")
        hyp = hyp2f1(r, b, a + b - 1, t / (alpha + t))

        return (a + b - 1) / (a - 1) * (1 - hyp * (alpha / (alpha + t)) ** r)

    def probability_of_n_purchases_up_to_time(
        self, 
        t, 
        n
    ):
        r"""
        Compute the probability of n purchases.

         .. math::  P( N(t) = n | \text{model} )

        where N(t) is the number of repeat purchases a customer makes in t
        units of time.

        Comes from equation (8) of [2]_.

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

        References
        ----------
        .. [2] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
        "Counting Your Customers the Easy Way: An Alternative to the
        Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
        """

        r, alpha, a, b = self._unload_params("r", "alpha", "a", "b")

        first_term = (
            beta(a, b + n)
            / beta(a, b)
            * gamma(r + n)
            / gamma(r)
            / gamma(n + 1)
            * (alpha / (alpha + t)) ** r
            * (t / (alpha + t)) ** n
        )

        if n > 0:
            j = np.arange(0, n)
            finite_sum = (gamma(r + j) / gamma(r) / gamma(j + 1) * (t / (alpha + t)) ** j).sum()
            second_term = beta(a + 1, b + n - 1) / beta(a, b) * (1 - (alpha / (alpha + t)) ** r * finite_sum)
        else:
            second_term = 0

        return first_term + second_term