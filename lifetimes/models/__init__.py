"""BaseInferencer class for all BTYD models.

This model defines the BaseInferencer and AbstractRFM abstract classes, which contain base methods shared by all BTYD models via inheritance, and an API boilerplate for all RFM models, respectively.

License:
    Copyright (c) 2022, Colt Allen

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import generator_stop
from __future__ import annotations

from abc import ABC, abstractmethod
import warnings
import json
import psutil

import numpy as np

import pymc as pm
import arviz as az
import aesara.tensor as at



from ..utils import ConvergenceError


class BaseInferencer:

    def __init__(self,model:pm.Model()=pm.Model()):

        self.model = model # Custom object class

    def __repr__(self) -> str:
        """Representation of BTYD model object."""
        classname = self.__class__.__name__
        try:
            row_str = f"estimated with {self.data.shape[0]} subjects."
        except AttributeError:
            row_str = ""

        try:
            param_str = self.params
            return f"<btyd.{classname}: {param_str} on {param_str} posterior parameters>"
  
        except AttributeError:
            return f"<btyd.{classname}>"

    def save_model(self, path):
        """
        Save InferenceData summary in JSON format.

        Parameters
        ----------
        path: str
            Path where to save model.

        """
        
        self.idata.to_json(path)

    def load_model(self, path):
        """
        Load posterior distributions of model parameters and estimation metadata from JSON.

        Parameters
        ----------
        path: str
            From what path load model.

        Returns
        -------
        posterior_dict : dict
            Posterior distributions of model parameters and other inference metadata.
        """
        with open(path, "rb") as arviz_json:
            self.posterior_dict = json.load(arviz_json)
        return self.posterior_dict
    
    @classmethod
    @abstractmethod
    def _loglike(cls,*args):
        """
        Log-likelihood function for model subclass. Must be constructed from aesara tensors.
        """
        pass
    
    def fit(self,rfm_df,model,tune=500,draws=1000,chains=psutil.Process().cpu_affinity()):
        """
        Save InferenceData summary in JSON format.

        Parameters
        ----------
        rfm_df: pandas.DataFrame
            Pandas dataframe containing customer ids, frequency, recency, T and monetary value columns.
        model: pymc.Model()
            Custom pymc model class used for inference.
        tune: int
            Number of samples for posterior parameter distribution convergence. These are discarded for inference.
        draws: int
            Number of samples from posterior parameter distrutions after tune period. These are retained for model usage.
        chain: int
            Number of sampling chains used for inference. It is best to set this to the number of CPU cores.
        
        Returns
        -------
        param_posteriors : dict
            Posterior distributions of model parameters and other inference metadata.
        
        """

        with model:
            idata = pm.sample(
                tune=tune,
                draws=draws,
                chains=4,
                target_accept=0.95,
                return_inferencedata=True
            )
        
        self.param_posteriors = idata
        return self.param_posteriors

    @staticmethod
    def _sample(array, n_samples):
        """Utility function for sampling from parameter posteriors."""
        idx = np.random.choice(np.arange(len(array)), n_samples, replace=True)
        return array[idx]
