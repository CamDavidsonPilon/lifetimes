from __future__ import generator_stop
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import warnings
import json
from typing import Union, Tuple, TypeVar, Generic

import numpy as np
import pandas as pd

import pymc as pm
import arviz as az
import aesara.tensor as at

from ..utils import _check_inputs


SELF = TypeVar("SELF")

class BaseModel(ABC, Generic[SELF]):

    @abstractmethod
    def __init__() -> None:
        """ self._param_list must be instantiated here, as well as model hyperpriors."""

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
            param_vals = np.around(self._unload_params(),decimals=1)
            param_str = str(dict(zip(self._param_list, param_vals)))
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
        
        Returns
        -------
        self
            with ``_idata`` attribute for model evaluation and predictions.
        """

        self.frequency, self.recency, self.T, self.monetary_value, _ = self._dataframe_parser(rfm_df)

        with self._model():
            self.idata = pm.sample(
                tune=tune,
                draws=draws,
                chains=4,
                cores=4, 
                target_accept=0.95,
                return_inferencedata=True
            )
        
        return self
        
    def save_model(self, filename: str) -> None:
        """
        Dump InferenceData from fitted model into a JSON file.

        Parameters
        ----------
        filename: str
            Path and/or filename where model will be saved.

        """
        
        self.idata.to_json(filename)

    def load_model(self, filename: str) -> SELF:
        """
        Load saved model JSON.

        Parameters
        ----------
        filename: str
            Path and/or filename of model JSON.

        Returns
        -------
        self
            with loaded ``_idata`` attribute for model evaluation and predictions.
        """

        self.idata = az.from_json(filename)
        
        # BETA TODO: Raise BTYDException.
        # if dict(filter(lambda item: self.__class__.__name__ not in item[0], self.idata.posterior.get('data_vars').items()))
            # raise BTYDException

        return self
    
    def _unload_params(self, posterior: bool = False) -> Union[Tuple[np.ndarray],Tuple[np.ndarray]]:
        """Extract parameter posteriors from _idata InferenceData attribute of fitted model."""

        if posterior:
            return tuple([self.idata.posterior.get(f'{self.__class__.__name__}::{var}').values for var in self._param_list])
        else:
            return tuple([self.idata.posterior.get(f'{self.__class__.__name__}::{var}').mean().to_numpy() for var in self._param_list])
    
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


class AliveAPI(ABC, Generic[SELF]):
    """
    Define predictive methods for all models except GammaGamma.
    In research literature these are commonly referred to as quantities of interest.
    """

    @abstractmethod
    def _conditional_probability_alive() -> None:
        pass
    
    @abstractmethod
    def _conditional_expected_number_of_purchases_up_to_time() -> None:
        pass
    
    @abstractmethod
    def _expected_number_of_purchases_up_to_time() -> None:
        pass
    
    @abstractmethod
    def _probability_of_n_purchases_up_to_time() -> None:
        pass
    
    quantities_of_interest = {
        'cond_prob_alive': _conditional_probability_alive,
        'cond_n_prchs_to_time': _conditional_expected_number_of_purchases_up_to_time,
        'n_prchs_to_time': _expected_number_of_purchases_up_to_time,
        'prob_n_prchs_to_time': _probability_of_n_purchases_up_to_time,
    }
