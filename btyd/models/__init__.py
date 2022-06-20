from __future__ import generator_stop
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import warnings
import json
import psutil
from typing import Dict, List, Iterable, TypeVar, Generic

import numpy as np
import pandas as pd

import pymc as pm
import arviz as az
import aesara.tensor as at

from ..utils import _check_inputs


SELF = TypeVar("SELF")

class BaseModel(ABC, Generic[SELF]):

    # This attribute must be defined in subclasses.
    remove_hypers: list

    @abstractmethod
    def _model() -> None:
        """ pymc model defining priors for model parameters and calling _loglike in Potential()."""
        pass

    @abstractmethod
    def _loglike() -> None:
        """ Log-likelihood function for randomly drawn individual from customer population. Must be constructed from aesara tensors. """
        pass
    
    @abstractmethod
    def predict() -> None:
        pass

    def __repr__(self) -> str:
        """Representation of BTYD model object."""
        classname = self.__class__.__name__
        try:
            row_str = f"estimated with {self.frequency.shape[0]} customers."
        except AttributeError:
            row_str = ""

        try:
            param_keys = [key.split(f'{classname}::')[1] for key in list(self.param_dict.get('data_vars').keys())]
            param_vals = np.around(self._unload_params(),decimals=1)
            param_str = str(dict(zip(param_keys, param_vals)))
            return f"<btyd.{classname}: Parameters {param_str} {row_str}>"
        except AttributeError:
            return f"<btyd.{classname}>"
    
    def fit(self, rfm_df: pd.DataFrame, tune: int = 1200, draws: int = 1200) -> SELF:
        """
        Fit a custom pymc model with parameter prior definitions to observed RFM data.

        Parameters
        ----------
        rfm_df: pandas.DataFrame
            Pandas dataframe containing customer ids, frequency, recency, T and monetary value columns.
        tune: int
            Number of beginning 'burn-in' samples for posterior parameter distribution convergence. These are discarded after model is fit.
        draws: int
            Number of samples from posterior parameter distrutions after tune period. These are retained for model usage.
        chain: int
            Number of sampling chains used for inference. It is best to set this to the number of CPU cores.
        
        Returns
        -------
        param_posteriors : dict
            Posterior distributions of model parameters and other inference metadata.
        
        """

        self.frequency, self.recency, self.T, self.monetary_value, _ = self._dataframe_parser(rfm_df)

        #self.model = self._model()

        with self._model():
            self.idata = pm.sample(
                tune=tune,
                draws=draws,
                chains=4,
                cores=len(psutil.Process().cpu_affinity()), 
                target_accept=0.95,
                return_inferencedata=True
            )
        
        self.param_dict = self.idata.posterior.to_dict()

        # Remove unneeded items from param_dict.
        del self.param_dict['coords']
        for var in self.remove_hypers:
            del self.param_dict.get('data_vars')[var]
        
        return self
        
    def save_params(self, path: str) -> None:
        """
        Save InferenceData object of posterior distributions for model parameters in JSON format.

        Parameters
        ----------
        path: str
            Filepath and/or name to save model.

        """
        
        with open(path, "w") as outfile:
            json.dump(self.param_dict, outfile)

    def load_params(self, path: str) -> Dict[str, float, int, List[List[float]], Tuple(str), dict]:
        """
        Load posterior distributions for model parameters and estimation metadata from JSON.

        Parameters
        ----------
        path: str
            From what path load model.

        Returns
        -------
        self.idata : dict
            Posterior distributions of model parameters and other inference metadata.
        """

        with open(path, "rb") as model_json:
            self.param_dict = json.load(model_json)
        
        # TODO: Raise BTYDException.
        # if dict(filter(lambda item: self.__class__.__name__ not in item[0], self.param_dict.get('data_vars').items()))
            # raise BTYDException

        return self.param_dict
    
    def _unload_params(self, posterior: bool = False) -> List[np.ndarray]:

        param_list = deepcopy(self.param_dict.get('data_vars'))

        for key in param_list:
            param_list[key]['data'] = np.array(param_list[key].get('data')).flatten()
            
        if not posterior:
            for key in param_list:
                param_list[key]['data'] = np.atleast_1d(param_list[key].get('data').mean())
                # param_list[key]['data'] = param_list[key].get('data').mean()

        return [param_list.get(var).get('data') for var in list(param_list.keys())]
    
    @staticmethod
    def _dataframe_parser(rfm_df: pd.DataFrame) -> Tuple[np.ndarray]:
        """ Parse input dataframe into separate RFM components. """

        rfm_df.columns = rfm_df.columns.str.upper()

        # The load_cdnow_summary_with_monetary_value() function needs an ID column for testing.
        if 'ID' not in rfm_df.columns:
            customer = rfm_df.index.values
        else:
            customer = rfm_df['ID'].values
        
        frequency = rfm_df['FREQUENCY'].values
        recency = rfm_df['RECENCY'].values
        T = rfm_df['T'].values
        monetary_value = rfm_df['MONETARY_VALUE'].values

        # TODO: Add monetary_value to this, and consider ID continengent on predict() outputs.
        _check_inputs(frequency, recency, T)

        return frequency, recency, T, monetary_value, customer

    @staticmethod
    def _sample(array: npt.ArrayLike, n_samples: int = 100) -> np.ndarray:
        """Utility function for sampling from parameter posteriors."""
        idx = np.random.choice(np.arange(len(array)), n_samples, replace=True)
        return array[idx]
