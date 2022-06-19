# -*- coding: utf-8 -*-
"""Gamma-Gamma Model."""

from __future__ import print_function
from __future__ import division
import warnings

import pandas as pd
from autograd import numpy as np
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

    def __init__(
        self, 
        penalizer_coef=0.0
    ):
        """
        Initialization, set penalizer_coef.
        """

        self.penalizer_coef = penalizer_coef

    @staticmethod
    def _negative_log_likelihood(
        log_params, 
        frequency, 
        avg_monetary_value, 
        weights, 
        penalizer_coef
    ):
        """
        Computes the Negative Log-Likelihood for the Gamma-Gamma Model as in:
        http://www.brucehardie.com/notes/025/

        This also applies a penalizer to the log-likelihood.

        Equivalent to equation (1a).

        Hardie's implementation of this method can be seen on page 8.
        """

        warnings.simplefilter(action="ignore", category=FutureWarning)

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
        ) * weights
        penalizer_term = penalizer_coef * sum(params ** 2)

        return -negative_log_likelihood_values.sum() / weights.sum() + penalizer_term

    def conditional_expected_average_profit(
        self, 
        frequency=None, 
        monetary_value=None
    ):
        """
        Conditional expectation of the average profit.

        This method computes the conditional expectation of the average profit
        per transaction for a group of one or more customers.

        Equation (5) from:
        http://www.brucehardie.com/notes/025/

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
        weights=None,
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
        weights: None or array_like
            Number of customers with given frequency/monetary_value,
            defaults to 1 if not specified. Fader and
            Hardie condense the individual RFM matrix into all
            observed combinations of frequency/monetary_value. This
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

        if weights is None:
            weights = np.ones_like(frequency, dtype=int)
        else:
            weights = np.asarray(weights)

        log_params, self._negative_log_likelihood_, self._hessian_ = self._fit(
            (frequency, monetary_value, weights, self.penalizer_coef),
            initial_params,
            3,
            verbose,
            tol=tol,
            bounds=((None, None), (0, None), (None, None)) if q_constraint else None,
            **kwargs
        )

        self.data = DataFrame(
            {"monetary_value": monetary_value, "frequency": frequency, "weights": weights}, index=index
        )

        self.params_ = pd.Series(np.exp(log_params), index=["p", "q", "v"])

        self.variance_matrix_ = self._compute_variance_matrix()
        self.standard_errors_ = self._compute_standard_errors()
        self.confidence_intervals_ = self._compute_confidence_intervals()

        return self

    def customer_lifetime_value(
        self, 
        transaction_prediction_model, 
        frequency, 
        recency, 
        T, 
        monetary_value, 
        time=12, 
        discount_rate=0.01, 
        freq="D"
    ):
        """
        Return customer lifetime value.

        This method computes the average lifetime value for a group of one
        or more customers.

        Parameters
        ----------
        transaction_prediction_model: model
            the model to predict future transactions, literature uses
            pareto/ndb models but we can also use a different model like beta-geo models
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
        freq: string, optional
            {"D", "H", "M", "W"} for day, hour, month, week. This represents what unit of time your T is measure in.

        Returns
        -------
        Series:
            Series object with customer ids as index and the estimated customer
            lifetime values as values
        """
        
        # use the Gamma-Gamma estimates for the monetary_values
        adjusted_monetary_value = self.conditional_expected_average_profit(frequency, monetary_value)

        return _customer_lifetime_value(
            transaction_prediction_model, frequency, recency, T, adjusted_monetary_value, time, discount_rate, freq=freq
        )
