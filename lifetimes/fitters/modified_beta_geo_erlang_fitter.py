# -*- coding: utf-8 -*-
"""MBG/CNBD-k model."""

from __future__ import print_function
from __future__ import division
import warnings

from math import factorial
from functools import partial
from multiprocessing.pool import Pool
import pandas as pd
import autograd.numpy as np
from autograd.scipy.special import gammaln, betaln
from scipy.special import hyp2f1
from scipy.stats import nbinom
from . import BaseFitter


class ModifiedBetaGeoErlangFitter(BaseFitter):
    """
    Also known as the (M)BG/CNBD-K model.

    Based on the BTYDplus implementation (in R)

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
    """

    def __init__(
        self,
        penalizer_coef=0.0,
        dropout_at_zero=True
    ):
        """
        Initialization, set penalizer_coef.
        """
        self.penalizer_coef = penalizer_coef
        self.dropout_at_zero = dropout_at_zero

    def fit_known_k(
            self,
            frequency,
            recency,
            T,
            litt,
            k,
            weights=None,
            initial_params=None,
            verbose=False,
            tol=1e-7,
            index=None,
            **kwargs
    ):
        frequency = np.asarray(frequency).astype(int)
        recency = np.asarray(recency)
        T = np.asarray(T)
        litt = np.asarray(litt)
        # _check_inputs(frequency, recency, T,litt) # whatever this means

        if weights is None:
            weights = np.ones_like(recency, dtype=int)
        else:
            weights = np.asarray(weights)

        # careful here

        log_params_, self._negative_log_likelihood_, self._hessian_ = self._fit(
            (k, frequency, recency, T, litt, True, weights, self.penalizer_coef),
            initial_params,
            4,
            verbose,
            tol,
            **kwargs
        )

        self.k_ = k # hm
        self.params_ = pd.Series(np.exp(log_params_), index=["r", "alpha", "a", "b"]) # hm

        self.data = pd.DataFrame({"frequency": frequency, "recency": recency, "T": T, "litt": litt, "weights": weights},
                                 index=index)

        self.generate_new_data = None  # not sure if I need it

        # self.predict = self.conditional_expected_number_of_purchases_up_to_time  # to be defined later

        #self.variance_matrix_ = self._compute_variance_matrix()  # whatever
        #self.standard_errors_ = self._compute_standard_errors()  # whatever
        #self.confidence_intervals_ = self._compute_confidence_intervals()  # whatever

        return self

    def fit(
        self,
        frequency,
        recency,
        T,
        litt,
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
        k: bla bla bla
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


        # _check_inputs(frequency, recency, T)

        l_mbgf = []
        l_ll = []
        for k in range(1, 13):
            try:
                mbgf = ModifiedBetaGeoErlangFitter(self.penalizer_coef)
                mbgf.fit_known_k(frequency, recency, T, litt, k, weights, initial_params, verbose, tol, index, **kwargs)
            except:
                break
            l_mbgf.append(mbgf)
            l_ll.append(mbgf._negative_log_likelihood_)
            if k > 2:
                if l_ll[k - 1] > l_ll[k - 2] > l_ll[k - 3]:
                    break

        mbgf = l_mbgf[l_ll.index(min(l_ll))]

        self.k_ = mbgf.k_
        self.params_ = mbgf.params_
        self.data = mbgf.data
        self._negative_log_likelihood_ = mbgf._negative_log_likelihood_
        self._hessian_ = mbgf._hessian_
        return self

    @staticmethod
    def _negative_log_likelihood(
        log_params,
        k,
        freq,
        rec,
        T,
        litt,
        dropout_at_zero, # True for the mbd
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

        P1 = (k - 1) * litt - freq * np.log(factorial(k - 1))
        P2 = betaln(a, b + freq + (1 if dropout_at_zero else 0)) - betaln(a, b)
        P3 = gammaln(r + k * freq) - gammaln(r) + r * np.log(alpha)
        P4 = -1 * (r + k * freq) * np.log(alpha + T)
        S1 = (int(dropout_at_zero or freq > 0) * a / (b + freq - 1 + (1 if dropout_at_zero else 0)) *
              ((alpha + T) / (alpha + rec)) ** (r + k * freq))

        S2 = 1
        if k > 1:
            for j in range(1,k):
                S2a = 1
                for i in range(j):
                    S2a = S2a * (r + k * freq + i)
                S2 = S2 + (S2a * (T - rec) ** j) / (factorial(j) * (alpha + T) ** j)

        penalizer_term = penalizer_coef * sum(params ** 2) # not taking k
        ll = weights * (P1 + P2 + P3 + P4 + np.log(S1 + S2))
        return -ll.sum() / weights.sum() + penalizer_term

    def conditional_expected_number_of_purchases_up_to_time(self, t, frequency, recency, T):
        k = self.k_
        r, alpha, a, b = self.params_
        dropout_at_zero = self.dropout_at_zero
        params = self.params_ # maybe should remove that...
        if round(a, 2) == 1:
            a = a+0.01

        P1 = (a + b + frequency - 1 + np.where(dropout_at_zero, 1, 0)) / (a - 1)
        P2 = self._compute_G(r + frequency, k * alpha + t, a, b + frequency - 1 + np.where(dropout_at_zero, 1, 0), T)
        P3 = self.compute_p_alive(frequency, recency, t)
        exp = P1 * P2 * P3
        correction_term = (1 if k==1 else self._compute_correction_term(t, T, exp, dropout_at_zero))
        return exp * correction_term

    @staticmethod
    def _compute_G(r, alpha, a, b, t):
        res = 1 - np.power((alpha / (alpha + t)), r) * hyp2f1(r, b + 1, a + b, t / (alpha + t))
        return res

    def compute_p_alive(self, frequency, recency, T):
        k = self.k_
        r, alpha, a, b = self.params_
        dropout_at_zero = self.dropout_at_zero

        P1 = (a / (b + frequency - 1 + (1 if dropout_at_zero else 0))) * np.power((alpha + T) / (alpha + recency), (r + k * frequency))
        P2 = self._compute_P2_alive(k, r, alpha, frequency, recency, T)

        p_alive = np.where((not dropout_at_zero) and frequency == 0, 1, 1 / (1 + P1 / P2))
        return p_alive

    @staticmethod
    def _compute_P2_alive(k, r, alpha, x, t_x, T_cal):
        P2 = 1
        if k > 1:
            for j in range(1,k):
                P2a = 1
                for i in range(j): P2a = P2a * (r + k * x + i)
                P2 = P2 + ((P2a * (T_cal - t_x) ** j) / (factorial(j) * (alpha + T_cal) ** j))
        return P2

    def _compute_correction_term(self, t_cal, t_star, exp, dropout_at_zero=False):
        sum_cal = np.sum(
            self.expected_purchases_avg_customer_list_t(t_cal, dropout_at_zero))
        sum_tot = np.sum(
            self.expected_purchases_avg_customer_list_t(t_cal + t_star, dropout_at_zero))
        corr_term = (sum_tot - sum_cal) / np.sum(exp)
        return corr_term

    def expected_purchases_avg_customer_list_t(self, t, dropout_at_zero = False):
        # we use parallelization
        k = self.k_
        params = self.params_
        min_t, max_t = (np.min(t),np.max(t))
        # we will usually need all values in the range
        dict_exp = self.expected_purchases_avg_customer_range_t(min_t, max_t, k, params, dropout_at_zero)
        res = self._vect_dict_lookup(dict_exp, t)
        return res

    @staticmethod
    def expected_purchases_avg_customer_range_t(min_t, max_t, k, params, dropout_at_zero = False):
        # we use parallelization
        exp_pur_k_params = partial(
            ModifiedBetaGeoErlangFitter.expected_purchases_avg_customer_single_t,
            k=k, params=params, dropout_at_zero = dropout_at_zero)
        p = Pool(8)
        res_keys = range(min_t, max_t+1)
        res_vals = p.map(exp_pur_k_params, res_keys)
        res_dict = dict(zip(res_keys,res_vals))
        return res_dict

    @staticmethod
    def _vect_dict_lookup(ref_dict, l_in):
        l_out = np.array([None]*len(l_in))
        for key, val in ref_dict.items():
            l_out[l_in == key] = val
        return l_out

    @staticmethod
    def expected_purchases_avg_customer_single_t(t, k, params, dropout_at_zero = False):
        res = 0
        r, alpha, a, b = params
        stop = max(nbinom.ppf(0.9999, n=r, p=alpha / (alpha+t)), 100) # vectorised!
        for i in range(1,int(stop)):
            add = i * ModifiedBetaGeoErlangFitter.xbgcnbd_pmf(k,params, t, i, dropout_at_zero)
            res+=add
            if (add < 1e-8) and (i>=100):
                break
        return res

    @staticmethod
    def xbgcnbd_pmf(k, params, t, x, dropout_at_zero = False):
        r, alpha, a,b = params
        survivals = x if dropout_at_zero else x-1
        if t == 0:
            return 0

        P1 = np.exp(gammaln(b + survivals + 1) + gammaln(a + b) - gammaln(b) - gammaln(a + b + survivals + 1))
        i = np.arange(k*x,k*(x+1))
        P2a = np.sum(np.exp(gammaln(r + i) + r * np.log(alpha) + i * np.log(t) - gammaln(i + 1) - gammaln(r) - (r + i) * np.log(alpha + t)))

        if (dropout_at_zero == False) and (x == 0):
            P2b = 0
        else:
            P2b = a / (b + survivals)
            if x>0:
                i = np.arange(k*x)
                cmf = np.sum(np.exp(
                    gammaln(r + i) + r * np.log(alpha) + i * np.log(t) - gammaln(i + 1) - gammaln(r) - (r + i) * np.log(alpha + t)
                ))
                P2b = P2b * (1 - cmf)
        res = P1 * (P2a + P2b)
        return res
