# -*- coding: utf-8 -*-
""" Beta Geo Fitter with Time-Invariant Covariates."""
from __future__ import print_function
from __future__ import division
from collections import OrderedDict

import autograd.numpy as np
import pandas as pd
from autograd.scipy.special import gammaln, beta, gamma, logsumexp
from scipy.special import hyp2f1

from . import BaseFitter
from ..utils import _scale_time, _check_inputs, _concat2
from ..generate_data import beta_geometric_nbd_model


class BetaGeoCovarsFitter(BaseFitter):
    """
    Also known as the BG/NBD model.
    Based on [2] and [3], this model has the following assumptions:
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
    params_: :obj: OrderedDict
        The fitted parameters of the model
    data: :obj: DataFrame
        A DataFrame with the columns given in the call to `fit`
    References
    ----------
    .. [2] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
       "Counting Your Customers the Easy Way: An Alternative to the
       Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
    .. [3] Peter S. Fader, Bruce G. S. Hardie, August 2007,
       Incorporating Time-Invariant Covariates into the Pareto/NBD and BG/NBD.
    """

    def __init__(
        self,
        penalizer_coef=0.0
    ):
        """Initialization, set penalizer_coef."""
        self.penalizer_coef = penalizer_coef

    def fit(
        self,
        frequency,
        recency,
        T,
        X_tr,
        X_do,
        weights=None,
        iterative_fitting=1,
        initial_params=None,
        verbose=False, tol=1e-4,
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
        X_tr: array_like
            n * d1 matrix containing covariates representing
            time-invariant user characteristics affecting the Transaction Rate.
            d1 as number of covariates and n as number of users.
        X_do: array_like
            n * d2 matrix containing covariates representing
            time-invariant user characteristics affecting the Drop Out.
            d2 as number of covariates and n as number of users.
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
        BetaGeoCovarsFitter
            with additional properties like ``params_`` and methods like ``predict``
        """
        frequency = np.asarray(frequency).astype(int)
        recency = np.asarray(recency)
        T = np.asarray(T)
        X_tr = np.asarray(X_tr)
        X_do = np.asarray(X_do)
        _check_inputs(frequency, recency, T)

        if weights is None:
            weights = np.ones_like(recency, dtype=np.int64)
        else:
            weights = np.asarray(weights)

        self._scale = _scale_time(T)
        scaled_recency = recency * self._scale
        scaled_T = T * self._scale

        d_tr_ = X_tr.shape[1]
        d_do_ = X_do.shape[1]

        log_params_, self._negative_log_likelihood_, self._hessian_ = self._fit(
            (frequency, scaled_recency, scaled_T, X_tr, X_do, weights, self.penalizer_coef),
            initial_params,
            4 + d_tr_ + d_do_ * 2,
            verbose,
            tol,
            **kwargs)

        params = np.exp(log_params_).tolist()
        params_dict = OrderedDict(zip(['r', 'alpha0', 'a0', 'b0', 'coefs_tr', 'coefs_do1', 'coefs_do2'],
                                       params[:4] +
                                       [params[4: 4 + d_tr_],
                                        params[4 + d_tr_: 4 + d_tr_ + d_do_],
                                        params[4 + d_tr_ + d_do_: 4 + d_tr_ + d_do_ * 2]]))
        self.params_ = pd.Series(params_dict)
        self.params_['alpha0'] /= self._scale

        self.data = pd.DataFrame(_concat2(frequency, recency, T, X_tr, X_do),
                                 columns=['frequency', 'recency', 'T'] +
                                         ['x_tr_' + str(d) for d in range(1, (d_tr_ + 1))] +
                                         ['x_do_' + str(d) for d in range(1, (d_do_ + 1))])
        if index is not None:
            self.data.index = index

        self.generate_new_data = lambda size=1: beta_geometric_nbd_model(
            T, *self._unload_params('r', 'alpha0', 'a0', 'b0'), size=size)

        self.predict = self.conditional_expected_number_of_purchases_up_to_time
        return self

    @staticmethod
    def _negative_log_likelihood(
        log_params,
        freq,
        rec,
        T,
        x_tr,
        x_do,
        weights,
        penalizer_coef
    ):
        params = np.exp(log_params)
        if np.any(params <= 0):
            return np.inf

        d_tr = x_tr.shape[1]
        d_do = x_do.shape[1]

        r, alpha0, a0, b0 = params[:4]
        coefs_tr = params[4: 4 + d_tr]
        coefs_do1 = params[4 + d_tr: 4 + d_tr + d_do]
        coefs_do2 = params[4 + d_tr + d_do: 4 + d_tr + d_do * 2]

        alpha = alpha0 * np.exp(- np.inner(x_tr, coefs_tr))
        a = a0 * np.exp(np.inner(x_do, coefs_do1))
        b = b0 * np.exp(np.inner(x_do, coefs_do2))

        A_1 = gammaln(r + freq) - gammaln(r) + r * np.log(alpha)
        A_2 = (gammaln(a + b) + gammaln(b + freq) - gammaln(b) -
               gammaln(a + b + freq))
        A_3 = -(r + freq) * np.log(alpha + T)

        d = _concat2(np.ones_like(freq), (freq > 0))
        A_4 = np.log(a) - np.log(b + np.where(freq == 0, 1, freq) - 1) - \
              (r + freq) * np.log(rec + alpha)

        A_4 = np.where(np.isnan(A_4) | np.isinf(A_4), 0, A_4)
        penalizer_term = penalizer_coef * np.sum(params ** 2)
        return - (weights * (A_1 + A_2 + logsumexp(_concat2(A_3, A_4), axis=1, b=d))).sum() / weights.sum() + penalizer_term

    def conditional_expected_number_of_purchases_up_to_time(
        self,
        t,
        frequency,
        recency,
        T,
        X_tr,
        X_do
    ):
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
        X_tr: array_like
            n * d1 matrix containing covariates representing
            time-invariant user characteristics affecting the Transaction Rate.
            d1 as number of covariates and n as number of users.
        X_do: array_like
            n * d2 matrix containing covariates representing
            time-invariant user characteristics affecting the Drop Out.
            d2 as number of covariates and n as number of users.
        Returns
        -------
        array_like
        """
        x = frequency
        r, alpha0, a0, b0, coefs_tr, coefs_do1, coefs_do2 = self._unload_params(
            'r', 'alpha0', 'a0', 'b0', 'coefs_tr', 'coefs_do1', 'coefs_do2')
        alpha = alpha0 * np.exp(- np.inner(X_tr, coefs_tr))
        a = a0 * np.exp(np.inner(X_do, coefs_do1))
        b = b0 * np.exp(np.inner(X_do, coefs_do2))

        _a = r + x
        _b = b + x
        _c = a + b + x - 1
        _z = t / (alpha + T + t)
        ln_hyp_term = np.log(hyp2f1(_a, _b, _c, _z))

        # if the value is inf, we are using a different but equivalent
        # formula to compute the function evaluation.
        ln_hyp_term_alt = np.log(hyp2f1(_c - _a, _c - _b, _c, _z)) + \
                          (_c - _a - _b) * np.log(1 - _z)
        ln_hyp_term = np.where(np.isinf(ln_hyp_term), ln_hyp_term_alt, ln_hyp_term)
        first_term = (a + b + x - 1) / (a - 1)
        second_term = (1 - np.exp(ln_hyp_term + (r + x) *
                               np.log((alpha + T) / (alpha + t + T))))

        numerator = first_term * second_term
        denominator = 1 + (x > 0) * (a / (b + x - 1)) * \
                      ((alpha + T) / (alpha + recency)) ** (r + x)

        return numerator / denominator

    def conditional_probability_alive(
        self,
        frequency,
        recency,
        T,
        X_tr,
        X_do,
        ln_exp_max=300
    ):
        """
        Compute conditional probability alive.
        Compute the probability that a customer with history
        (frequency, recency, T) is currently alive.
        From http://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf
        Parameters
        ----------
        frequency: float
            historical frequency of customer.
        recency: float
            historical recency of customer.
        T: float
            age of the customer.
        X_tr: array_like
            n * d1 matrix containing covariates representing
            time-invariant user characteristics affecting the Transaction Rate.
            d1 as number of covariates and n as number of users.
        X_do: array_like
            n * d2 matrix containing covariates representing
            time-invariant user characteristics affecting the Drop Out.
            d2 as number of covariates and n as number of users.
        ln_exp_max: int
            to what value clip log_div equation
        Returns
        -------
        float
            value representing a probability
        """
        r, alpha0, a0, b0, coefs_tr, coefs_do1, coefs_do2 = self._unload_params(
            'r', 'alpha0', 'a0', 'b0', 'coefs_tr', 'coefs_do1', 'coefs_do2')
        alpha = alpha0 * np.exp(- np.inner(X_tr, coefs_tr))
        a = a0 * np.exp(np.inner(X_do, coefs_do1))
        b = b0 * np.exp(np.inner(X_do, coefs_do2))

        log_div = (r + frequency) * np.log(
            (alpha + T) / (alpha + recency)) + np.log(
            a / (b + np.where(frequency == 0, 1, frequency) - 1))

        return np.where(frequency == 0, 1.,
                     np.where(log_div > ln_exp_max, 0.,
                           1. / (1 + np.exp(np.clip(log_div, None, ln_exp_max)))))

    def conditional_probability_alive_matrix(
        self,
        X_tr,
        X_do,
        max_frequency=None,
        max_recency=None
    ):
        """
        Compute the probability alive matrix.
        Parameters
        ----------
        X_tr: array_like
            n * d1 matrix containing covariates representing
            time-invariant user characteristics affecting the Transaction Rate.
            d1 as number of covariates and n as number of users.
        X_do: array_like
            n * d2 matrix containing covariates representing
            time-invariant user characteristics affecting the Drop Out.
            d2 as number of covariates and n as number of users.
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

        return np.fromfunction(self.conditional_probability_alive,
                               (max_frequency + 1, max_recency + 1),
                               T=max_recency,
                               X_tr=1,
                               X_do=1,
                               ).T

    def expected_number_of_purchases_up_to_time(
        self,
        t,
        X_tr,
        X_do
    ):
        """
        Calculate the expected number of repeat purchases up to time t.
        Calculate repeat purchases for a randomly choose individual from the
        population.
        Parameters
        ----------
        t: array_like
            times to calculate the expection for
        X_tr: array_like
            n * d1 matrix containing covariates representing
            time-invariant user characteristics affecting the Transaction Rate.
            d1 as number of covariates and n as number of users.
        X_do: array_like
            n * d2 matrix containing covariates representing
            time-invariant user characteristics affecting the Drop Out.
            d2 as number of covariates and n as number of users.
        Returns
        -------
        array_like
        """
        r, alpha0, a0, b0, coefs_tr, coefs_do1, coefs_do2 = self._unload_params(
            'r', 'alpha0', 'a0', 'b0', 'coefs_tr', 'coefs_do1', 'coefs_do2')
        alpha = alpha0 * np.exp(- np.inner(X_tr, coefs_tr))
        a = a0 * np.exp(np.inner(X_do, coefs_do1))
        b = b0 * np.exp(np.inner(X_do, coefs_do2))

        hyp = hyp2f1(r, b, a + b - 1, t / (alpha + t))
        return (a + b - 1) / (a - 1) * (1 - hyp * (alpha / (alpha + t)) ** r)

    def probability_of_n_purchases_up_to_time(
        self,
        t,
        n,
        X_tr,
        X_do
    ):
        r"""
        Compute the probability of n purchases.
         .. math::  P( N(t) = n | \text{model} )
        where N(t) is the number of repeat purchases a customer makes in t
        units of time.
        Parameters
        ----------
        t: float
            number units of time
        n: int
            number of purchases
        X_tr: array_like
            n * d1 matrix containing covariates representing
            time-invariant user characteristics affecting the Transaction Rate.
            d1 as number of covariates and n as number of users.
        X_do: array_like
            n * d2 matrix containing covariates representing
            time-invariant user characteristics affecting the Drop Out.
            d2 as number of covariates and n as number of users.
        Returns
        -------
        float:
            Probability to have n purchases up to t units of time
        """
        r, alpha0, a0, b0, coefs_tr, coefs_do1, coefs_do2 = self._unload_params(
            'r', 'alpha0', 'a0', 'b0', 'coefs_tr', 'coefs_do1', 'coefs_do2')
        alpha = alpha0 * np.exp(- np.inner(X_tr, coefs_tr))
        a = a0 * np.exp(np.inner(X_do, coefs_do1))
        b = b0 * np.exp(np.inner(X_do, coefs_do2))

        first_term = (beta(a, b + n) / beta(a, b) *
                      gamma(r + n) / gamma(r) /
                      gamma(n + 1) * (alpha / (alpha + t)) ** r *
                      (t / (alpha + t)) ** n)

        if n > 0:
            j = np.arange(0, n)
            finite_sum = (gamma(r + j) / gamma(r) / gamma(j + 1) *
                          (t / (alpha + t)) ** j).sum()
            second_term = (beta(a + 1, b + n - 1) /
                           beta(a, b) * (1 - (alpha / (alpha + t)) ** r *
                                         finite_sum))
        else:
            second_term = 0
        return first_term + second_term
