# -*- coding: utf-8 -*-
"""Beta Geo Beta BinomFitter."""
from __future__ import division
from __future__ import print_function
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from autograd.numpy import log, exp, logaddexp
from pandas import DataFrame
from autograd.scipy.special import gammaln, betaln, beta as betaf
from scipy.special import binom

from ..utils import _check_inputs
from . import BaseFitter
from ..generate_data import beta_geometric_beta_binom_model


class BetaGeoBetaBinomFitter(BaseFitter):
    """
    Also known as the Beta-Geometric/Beta-Binomial Model [1]_.

    Future purchases opportunities are treated as discrete points in time.
    In the literature, the model provides a better fit than the Pareto/NBD
    model for a nonprofit organization with regular giving patterns.

    The model is estimated with a recency-frequency matrix with n transaction
    opportunities.

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
    .. [1] Fader, Peter S., Bruce G.S. Hardie, and Jen Shang (2010),
       "Customer-Base Analysis in a Discrete-Time Noncontractual Setting,"
       Marketing Science, 29 (6), 1086-1108.

    """

    def __init__(self, penalizer_coef=0.0):
        """Initialization, set penalizer_coef."""
        self.penalizer_coef = penalizer_coef

    @staticmethod
    def _loglikelihood(params, x, tx, T):
        warnings.simplefilter(action="ignore", category=FutureWarning)

        """Log likelihood for optimizer."""
        alpha, beta, gamma, delta = params

        betaln_ab = betaln(alpha, beta)
        betaln_gd = betaln(gamma, delta)

        A = betaln(alpha + x, beta + T - x) - betaln_ab + betaln(gamma, delta + T) - betaln_gd

        B = 1e-15 * np.ones_like(T)
        recency_T = T - tx - 1

        for j in np.arange(recency_T.max() + 1):
            ix = recency_T >= j
            B = B + ix * betaf(alpha + x, beta + tx - x + j) * betaf(gamma + 1, delta + tx + j)

        B = log(B) - betaln_gd - betaln_ab
        return logaddexp(A, B)

    @staticmethod
    def _negative_log_likelihood(log_params, frequency, recency, n_periods, weights, penalizer_coef=0):
        params = exp(log_params)
        penalizer_term = penalizer_coef * sum(params ** 2)
        return (
            -(BetaGeoBetaBinomFitter._loglikelihood(params, frequency, recency, n_periods) * weights).sum()
            / weights.sum()
            + penalizer_term
        )

    def fit(
        self,
        frequency,
        recency,
        n_periods,
        weights=None,
        initial_params=None,
        verbose=False,
        tol=1e-7,
        index=None,
        **kwargs
    ):
        """
        Fit the BG/BB model.

        Parameters
        ----------
        frequency: array_like
            Total periods with observed transactions
        recency: array_like
            Period of most recent transaction
        n_periods: array_like
            Number of transaction opportunities. Previously called `n`.
        weights: None or array_like
            Number of customers with given frequency/recency/T,
            defaults to 1 if not specified. Fader and
            Hardie condense the individual RFM matrix into all
            observed combinations of frequency/recency/T. This
            parameter represents the count of customers with a given
            purchase pattern. Instead of calculating individual
            log-likelihood, the log-likelihood is calculated for each
            pattern and multiplied by the number of customers with
            that pattern.  Previously called `n_custs`.
        verbose: boolean, optional
            Set to true to print out convergence diagnostics.
        tol: float, optional
            Tolerance for termination of the function minimization process.
        index: array_like, optional
            Index for resulted DataFrame which is accessible via self.data
        kwargs:
            Key word arguments to pass to the scipy.optimize.minimize
            function as options dict

        Returns
        -------
        BetaGeoBetaBinomFitter
            fitted and with parameters estimated

        """
        frequency = np.asarray(frequency).astype(int)
        recency = np.asarray(recency).astype(int)
        n_periods = np.asarray(n_periods).astype(int)

        if weights is None:
            weights = np.ones_like(recency)
        else:
            weights = np.asarray(weights)

        _check_inputs(frequency, recency, n_periods)

        log_params_, self._negative_log_likelihood_, self._hessian_ = self._fit(
            (frequency, recency, n_periods, weights, self.penalizer_coef), initial_params, 4, verbose, tol, **kwargs
        )
        self.params_ = pd.Series(np.exp(log_params_), index=["alpha", "beta", "gamma", "delta"])

        self.data = DataFrame(
            {"frequency": frequency, "recency": recency, "n_periods": n_periods, "weights": weights}, index=index
        )

        self.generate_new_data = lambda size=1: beta_geometric_beta_binom_model(
            # Making a large array replicating n by n_custs having n.
            np.array(sum([n_] * n_cust for (n_, n_cust) in zip(n_periods, weights))),
            *self._unload_params("alpha", "beta", "gamma", "delta"),
            size=size
        )

        self.variance_matrix_ = self._compute_variance_matrix()
        self.standard_errors_ = self._compute_standard_errors()
        self.confidence_intervals_ = self._compute_confidence_intervals()
        return self

    def conditional_expected_number_of_purchases_up_to_time(self, m_periods_in_future, frequency, recency, n_periods):
        r"""
        Conditional expected purchases in future time period.

        The  expected  number  of  future  transactions across the next m_periods_in_future
        transaction opportunities by a customer with purchase history
        (x, tx, n).

        .. math:: E(X(n_{periods}, n_{periods}+m_{periods_in_future})| \alpha, \beta, \gamma, \delta, frequency, recency, n_{periods})

        See (13) in Fader & Hardie 2010.

        Parameters
        ----------
        t: array_like
            time n_periods (n+t)

        Returns
        -------
        array_like
            predicted transactions

        """
        x = frequency
        tx = recency
        n = n_periods

        params = self._unload_params("alpha", "beta", "gamma", "delta")
        alpha, beta, gamma, delta = params

        p1 = 1 / exp(self._loglikelihood(params, x, tx, n))
        p2 = exp(betaln(alpha + x + 1, beta + n - x) - betaln(alpha, beta))
        p3 = delta / (gamma - 1) * exp(gammaln(gamma + delta) - gammaln(1 + delta))
        p4 = exp(gammaln(1 + delta + n) - gammaln(gamma + delta + n))
        p5 = exp(gammaln(1 + delta + n + m_periods_in_future) - gammaln(gamma + delta + n + m_periods_in_future))

        return p1 * p2 * p3 * (p4 - p5)

    def conditional_probability_alive(self, m_periods_in_future, frequency, recency, n_periods):
        """
        Conditional probability alive.

        Conditional probability customer is alive at transaction opportunity
        n_periods + m_periods_in_future.

        .. math:: P(alive at n_periods + m_periods_in_future|alpha, beta, gamma, delta, frequency, recency, n_periods)

        See (A10) in Fader and Hardie 2010.

        Parameters
        ----------
        m: array_like
            transaction opportunities

        Returns
        -------
        array_like
            alive probabilities

        """
        params = self._unload_params("alpha", "beta", "gamma", "delta")
        alpha, beta, gamma, delta = params

        p1 = betaln(alpha + frequency, beta + n_periods - frequency) - betaln(alpha, beta)
        p2 = betaln(gamma, delta + n_periods + m_periods_in_future) - betaln(gamma, delta)
        p3 = self._loglikelihood(params, frequency, recency, n_periods)

        return exp(p1 + p2) / exp(p3)

    def expected_number_of_transactions_in_first_n_periods(self, n):
        r"""
        Return expected number of transactions in first n n_periods.

        Expected number of transactions occurring across first n transaction
        opportunities.
        Used by Fader and Hardie to assess in-sample fit.

        .. math:: Pr(X(n) = x| \alpha, \beta, \gamma, \delta)

        See (7) in Fader & Hardie 2010.

        Parameters
        ----------
        n: float
            number of transaction opportunities

        Returns
        -------
        DataFrame:
            Predicted values, indexed by x

        """
        params = self._unload_params("alpha", "beta", "gamma", "delta")
        alpha, beta, gamma, delta = params

        x_counts = self.data.groupby("frequency")["weights"].sum()
        x = np.asarray(x_counts.index)

        p1 = binom(n, x) * exp(
            betaln(alpha + x, beta + n - x) - betaln(alpha, beta) + betaln(gamma, delta + n) - betaln(gamma, delta)
        )

        I = np.arange(x.min(), n)

        @np.vectorize
        def p2(j, x):
            i = I[int(j) :]
            return np.sum(
                binom(i, x)
                * exp(
                    betaln(alpha + x, beta + i - x)
                    - betaln(alpha, beta)
                    + betaln(gamma + 1, delta + i)
                    - betaln(gamma, delta)
                )
            )

        p1 += np.fromfunction(p2, (x.shape[0],), x=x)

        idx = pd.Index(x, name="frequency")
        return DataFrame(p1 * x_counts.sum(), index=idx, columns=["model"])
