# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import pandas as pd
from autograd import numpy as np
from autograd.numpy import c_ as vconcat
from pandas import DataFrame
from autograd.scipy.special import gammaln


from . import BaseFitter
from ..utils import _check_inputs, _customer_lifetime_value


class GammaGammaFitter(BaseFitter):
    """
    Fitter for the gamma-gamma model.

    It is used to estimate the average monetary value of customer transactions.

    This implementation is based on the Excel spreadsheet found in [3]_.
    More details on the derivation and evaluation can be found in [4]_.

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
    .. [3] http://www.brucehardie.com/notes/025/
       The Gamma-Gamma Model of Monetary Value.
    .. [4] Peter S. Fader, Bruce G. S. Hardie, and Ka Lok Lee (2005),
       "RFM and CLV: Using iso-value curves for customer base analysis",
       Journal of Marketing Research, 42 (November), 415-430.

    """

    def __init__(self, penalizer_coef=0.0):
        """Initialization, set penalizer_coef."""
        self.penalizer_coef = penalizer_coef

    @staticmethod
    def _negative_log_likelihood(log_params, frequency, avg_monetary_value):

        params = np.exp(log_params)
        p, q, v = params

        x = frequency
        m = avg_monetary_value

        negative_log_likelihood_values = (
            gammaln(p * x + q)
            - gammaln(p * x)
            - gammaln(q)
            + q * np.log(v)
            + (p * x - 1) * np.log(m)
            + (p * x) * np.log(x)
            - (p * x + q) * np.log(x * m + v)
        )
        penalizer_term = penalizer_coef * sum(params ** 2)
        return -np.mean(negative_log_likelihood_values) + penalizer_term

    def conditional_expected_average_profit(self, frequency=None, monetary_value=None):
        """
        Conditional expectation of the average profit.

        This method computes the conditional expectation of the average profit
        per transaction for a group of one or more customers.

        Parameters
        ----------
        frequency: array_like, optional
            a vector containing the customers' frequencies.
            Defaults to the whole set of frequencies used for fitting the model.
        monetary_value: array_like, optional
            a vector containing the customers' monetary values.
            Defaults to the whole set of monetary values used for
            fitting the model.

        Returns
        -------
        array_like:
            The conditional expectation of the average profit per transaction

        """
        if monetary_value is None:
            monetary_value = self.data["monetary_value"]
        if frequency is None:
            frequency = self.data["frequency"]
        p, q, v = self._unload_params("p", "q", "v")

        # The expected average profit is a weighted average of individual
        # monetary value and the population mean.
        individual_weight = p * frequency / (p * frequency + q - 1)
        population_mean = v * p / (q - 1)
        return (1 - individual_weight) * population_mean + individual_weight * monetary_value

    def fit(
        self,
        frequency,
        monetary_value,
        initial_params=None,
        verbose=False,
        tol=1e-7,
        index=None,
        q_constraint=False,
        **kwargs
    ):
        """
        Fit the data to the Gamma/Gamma model.

        Parameters
        ----------
        frequency: array_like
            the frequency vector of customers' purchases
            (denoted x in literature).
        monetary_value: array_like
            the monetary value vector of customer's purchases
            (denoted m in literature).
        initial_params: array_like, optional
            set the initial parameters for the fitter.
        verbose : bool, optional
            set to true to print out convergence diagnostics.
        tol : float, optional
            tolerance for termination of the function minimization process.
        index: array_like, optional
            index for resulted DataFrame which is accessible via self.data
        q_constraint: bool, optional
            when q < 1, population mean will result in a negative value
            leading to negative CLV outputs. If True, we penalize negative values of q to avoid this issue.
        kwargs:
            key word arguments to pass to the scipy.optimize.minimize
            function as options dict

        Returns
        -------
        GammaGammaFitter
            fitted and with parameters estimated

        """
        _check_inputs(frequency, monetary_value=monetary_value)

        frequency = np.asarray(frequency).astype(float)
        monetary_value = np.asarray(monetary_value).astype(float)

        log_params, self._negative_log_likelihood_, self._hessian_ = self._fit(
            (frequency, monetary_value, self.penalizer_coef),
            initial_params,
            3,
            verbose,
            tol=tol,
            bounds=((None, None), (0, None), (None, None)) if q_constraint else None,
            **kwargs
        )

        self.data = DataFrame(vconcat[frequency, monetary_value], columns=["frequency", "monetary_value"])
        if index is not None:
            self.data.index = index
        self.params_ = pd.Series(np.exp(log_params), index=["p", "q", "v"])

        return self

    def customer_lifetime_value(
        self, transaction_prediction_model, frequency, recency, T, monetary_value, time=12, discount_rate=0.01
    ):
        """
        Return customer lifetime value.

        This method computes the average lifetime value for a group of one
        or more customers.

        Parameters
        ----------
        transaction_prediction_model: model
            the model to predict future transactions, literature uses
            pareto/ndb but we can also use a different model like bg
        frequency: array_like
            the frequency vector of customers' purchases
            (denoted x in literature).
        recency: the recency vector of customers' purchases
                 (denoted t_x in literature).
        T: array_like
            customers' age (time units since first purchase)
        monetary_value: array_like
            the monetary value vector of customer's purchases
            (denoted m in literature).
        time: float, optional
            the lifetime expected for the user in months. Default: 12
        discount_rate: float, optional
            the monthly adjusted discount rate. Default: 0.01

        Returns
        -------
        Series:
            Series object with customer ids as index and the estimated customer
            lifetime values as values

        """
        # use the Gamma-Gamma estimates for the monetary_values
        adjusted_monetary_value = self.conditional_expected_average_profit(frequency, monetary_value)
        return _customer_lifetime_value(
            transaction_prediction_model, frequency, recency, T, adjusted_monetary_value, time, discount_rate
        )
