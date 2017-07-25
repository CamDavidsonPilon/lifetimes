"""Beta Geo Beta BinomFitter."""

from __future__ import print_function
from collections import OrderedDict

import numpy as np
import pandas as pd
from numpy import log, exp, logaddexp, asarray, c_ as vconcat
from pandas import DataFrame
from scipy.special import gammaln, betaln, binom

from ..utils import _fit, _check_inputs
from . import BaseFitter


class BetaGeoBetaBinomFitter(BaseFitter):
    """Also known as the Beta-Geometric/Beta-Binomial Model [1]_.

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
    params_: :obj: OrderedDict
        The fitted parameters of the model
    data: :obj: DataFrame
        A DataFrame with the columns given in the call to `fit`

    References
    ----------
    .. [1] Fader, Peter S., Bruce G.S. Hardie, and Jen Shang (2010),
       "Customer-Base Analysis in a Discrete-Time Noncontractual Setting,"
       Marketing Science, 29 (6), 1086-1108.
    """

    def __init__(self, penalizer_coef=0.):
        """Initialization, set penalizer_coef."""
        self.penalizer_coef = penalizer_coef

    @staticmethod
    def _loglikelihood(params, x, tx, T):
        """Loglikelihood for optimizer."""
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
    def _negative_log_likelihood(params, frequency, recency, n, n_custs,
                                 penalizer_coef=0):
        penalizer_term = penalizer_coef * sum(np.asarray(params) ** 2)
        return -np.mean(BetaGeoBetaBinomFitter._loglikelihood(
            params, frequency, recency, n) * n_custs) + penalizer_term

    def fit(self, frequency, recency, n, n_custs, verbose=False,
            tol=1e-4, iterative_fitting=1, index=None,
            fit_method='Nelder-Mead', maxiter=2000, **kwargs):
        """
        Fit the BG/BB model.

        Parameters
        ----------
        frequency: array_like
            Total periods with observed transactions
        recency: array_like
            Period of most recent transaction
        n: array_like
            Number of transaction opportunities.
        n_custs: array_like
            Number of customers with given frequency/recency/T. Fader
            and Hardie condense the individual RFM matrix into all
            observed combinations of frequency/recency/T. This
            parameter represents the count of customers with a given
            purchase pattern. Instead of calculating individual
            loglikelihood, the loglikelihood is calculated for each
            pattern and multiplied by the number of customers with
            that pattern.
        verbose: boolean, optional
            Set to true to print out convergence diagnostics.
        tol: float, optional
            Tolerance for termination of the function minimization process.
        iterative_fitting: int, optional
            perform iterative_fitting fits over random/warm-started initial params
        index: array_like, optional
            Index for resulted DataFrame which is accessible via self.data
        fit_method: string, optional
            Fit_method to passing to scipy.optimize.minimize
        maxiter: int, optional
            Max iterations for optimizer in scipy.optimize.minimize
            will be overwritten if setted in kwargs.
        kwargs:
            Key word arguments to pass to the scipy.optimize.minimize
            function as options dict

        Returns
        -------
        BetaGeoBetaBinomFitter
            fitted and with parameters estimated

        """
        frequency = asarray(frequency)
        recency = asarray(recency)
        n = asarray(n)
        n_custs = asarray(n_custs)
        _check_inputs(frequency, recency, n)

        params, self._negative_log_likelihood_ = _fit(
            self._negative_log_likelihood,
            [frequency, recency, n, n_custs, self.penalizer_coef],
            iterative_fitting,
            np.ones(4),
            4,
            verbose,
            tol,
            fit_method,
            maxiter,
            **kwargs)
        self.params_ = OrderedDict(zip(['alpha', 'beta', 'gamma', 'delta'],
                                       params))
        self.data = DataFrame(vconcat[frequency, recency, n, n_custs],
                              columns=['frequency', 'recency', 'n', 'n_custs'])
        if index is not None:
            self.data.index = index

        return self

    def conditional_expected_number_of_purchases_up_to_time(self, t):
        """
        Conditional expected purchases in future time period.

        The  expected  number  of  future  transactions across the next t
        transaction opportunities by a customer with purchase history
        (x, tx, n).

        .. math:: E(X(n, n+n*)|alpha, beta, gamma, delta, frequency, recency, n)

        See (13) in Fader & Hardie 2010.

        Parameters
        ----------
        t: array_like
            time periods (n+t)

        Returns
        ----------
        array_like
            predicted transactions

        """
        x = self.data['frequency']
        tx = self.data['recency']
        n = self.data['n']

        params = self._unload_params('alpha', 'beta', 'gamma', 'delta')
        alpha, beta, gamma, delta = params

        p1 = 1 / exp(self._loglikelihood(params, x, tx, n))
        p2 = exp(betaln(alpha + x + 1, beta + n - x) - betaln(alpha, beta))
        p3 = delta / (gamma - 1) * exp(gammaln(gamma + delta) -
                                       gammaln(1 + delta))
        p4 = exp(gammaln(1 + delta + n) - gammaln(gamma + delta + n))
        p5 = exp(gammaln(1 + delta + n + t) - gammaln(gamma + delta + n + t))

        return p1 * p2 * p3 * (p4 - p5)

    def conditional_probability_alive(self, m):
        """
        Conditional probability alive.

        Conditional probability customer is alive at transaction opportunity
        n + m.

        .. math:: P(alive at n + m|alpha, beta, gamma, delta, frequency, recency, n)

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
        params = self._unload_params('alpha', 'beta', 'gamma', 'delta')
        alpha, beta, gamma, delta = params

        x = self.data['frequency']
        tx = self.data['recency']
        n = self.data['n']

        p1 = betaln(alpha + x, beta + n - x) - betaln(alpha, beta)
        p2 = betaln(gamma, delta + n + m) - betaln(gamma, delta)
        p3 = self._loglikelihood(params, x, tx, n)

        return exp(p1 + p2) / exp(p3)

    def expected_number_of_transactions_in_first_n_periods(self, n):
        """
        Return expected number of transactions in first n periods.

        Expected number of transactions occurring across first n transaction
        opportunities.
        Used by Fader and Hardie to assess in-sample fit.

        .. math:: Pr(X(n) = x|alpha, beta, gamma, delta)

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
        params = self._unload_params('alpha', 'beta', 'gamma', 'delta')
        alpha, beta, gamma, delta = params

        x_counts = self.data.groupby('frequency')['n_custs'].sum()
        x = asarray(x_counts.index)

        p1 = binom(n, x) * exp(betaln(alpha + x, beta + n - x) -
                               betaln(alpha, beta) +
                               betaln(gamma, delta + n) -
                               betaln(gamma, delta))

        I = np.arange(x.min(), n)

        @np.vectorize
        def p2(j, x):
            i = I[int(j):]
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
