"""Module for Beta-Geo/Beta Binomial Model.

Usage:
    BetaGeo(params).fit()

License:
    Copyright (c) 2022, Colt Allen

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import generator_stop
from __future__ import annotations
import warnings

import pandas as pd
import numpy as np

import pymc as pm
import aesara.tensor as at

from . import BaseModel
from ..utils import _scale_time, _check_inputs
from ..generate_data import beta_geometric_nbd_model


class BetaGeoModel(BaseModel):
    """
    Also known as the BG/NBD model.
    Based on [2]_, this model has the following assumptions:
    1) Each individual, i, has a hidden lambda_i and p_i parameter
    2) These come from a population wide Gamma and a Beta distribution
       respectively.
    3) Individuals purchases follow a Poisson process with rate lambda_i*t .
    4) After each purchase, an individual has a p_i probability of dieing
       (never buying again).

    Attributes
    ----------
    params_: :obj: Series
        The fitted parameters of the model
    data: :obj: DataFrame
        A DataFrame with the values given in the call to `fit`
    attr_name : datatype
        Add description here.
        
    References
    ----------
    .. [2] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
       "Counting Your Customers the Easy Way: An Alternative to the
       Pareto/NBD Model," Marketing Science, 24 (2), 275-84.

    """

    def __init__(
        self, 
        a,b,alpha,r
    ):

    def logp_full(rfm_df, lam, p):
        """Individual-level log-likelihood function of full posterior distribution for each customer.

        The following method for calculatating the *log-likelihood* is derived from equation 3
        specified in section 4 of [2]_. More information can also be found in [3]_.

        References
        ----------
        .. [2] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
        "Counting Your Customers the Easy Way: An Alternative to the
        Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
        .. [3] http://brucehardie.com/notes/004/
        """
        
        n = rfm_df.shape[0]
        x = rfm_df["frequency"].to_numpy()
        t_x = rfm_df["recency"].to_numpy()
        T = rfm_df["T"].to_numpy()
        
        # Flag instances where frequency > 0 for switch statement.
        int_vec = np.vectorize(int)
        x_zero = int_vec(x > 0)

        log_term_a = x * at.log(1 - p) + x * at.log(lam) - t_x * lam
        term_b_1 = -lam * (T - t_x)
        term_b_2 = at.log(p) - at.log(1 - p)
        log_term_b = pm.math.switch(x_zero, pm.math.logaddexp(term_b_1, term_b_2), term_b_1)

        return at.sum(log_term_a) + at.sum(log_term_b)

    def logp(rfm_df, a, b, alpha, r):
        """Log-likelihood function of posterior distributions for model parameters.

        The following method for calculatating the *log-likelihood* is derived from the method
        specified in section 7 of [2]_. More information can also be found in [3]_.

        References
        ----------
        .. [2] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
        "Counting Your Customers the Easy Way: An Alternative to the
        Pareto/NBD Model," Marketing Science, 24 (2), 275-84.
        .. [3] http://brucehardie.com/notes/004/
        """

        n = rfm_df.shape[0]
        x = rfm_df["frequency"].to_numpy()
        t_x = rfm_df["recency"].to_numpy()
        T = rfm_df["T"].to_numpy()
        
        # Flag instances where frequency > 0 for switch statement.
        int_vec = np.vectorize(int)
        x_zero = int_vec(x > 0)

        a1 = at.gammaln(r + x) - at.gammaln(r) + r * at.log(alpha)
        a2 = at.gammaln(a + b) + at.gammaln(b + x) - at.gammaln(b) - at.gammaln(a + b + x)
        a3 = -(r + x) * at.log(alpha + T)
        a4 =  at.log(a) - at.log(b + at.maximum(x, 1) - 1) - (r + x) * at.log(t_x + alpha)
        max_a3_a4 = at.maximum(a3, a4)
        ll_1 = a1 + a2 
        ll_2 = at.log(at.exp(a3 - max_a3_a4) + at.exp(a4 - max_a3_a4) * pm.math.switch(x_zero, 1, 0)) + max_a3_a4

        return at.sum(ll_1 + ll_2)