from __future__ import generator_stop
from __future__ import annotations

from abc import ABC, abstractmethod
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

    # This attribute must be defined in model subclasses.
    _quantities_of_interest: dict

    @abstractmethod
    def __init__(self) -> SELF:
        """ self._param_list must be instantiated here, as well as model hyperpriors."""

    @abstractmethod
    def _model(self) -> None:
        """ pymc model defining priors for model parameters and calling _log_likelihood in Potential()."""
        pass

    @abstractmethod
    def _log_likelihood(self) -> None:
        """ Log-likelihood function for randomly drawn individual from customer population. Must be constructed from aesara tensors. """
        pass

    @abstractmethod
    def generate_rfm_data(self) -> None:
        pass

    def __repr__(self) -> str:
        """Representation of BTYD model object."""
        classname = self.__class__.__name__
        try:
            row_str = f"estimated with {self._frequency.shape[0]} customers."
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

        self._frequency, self._recency, self._T, self._monetary_value, _ = self._dataframe_parser(rfm_df)

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
    
    def _unload_params(self, posterior: bool = False, n_samples: int = 100) -> Union[Tuple[np.ndarray],Tuple[np.ndarray]]:
        """Extract parameter posteriors from _idata InferenceData attribute of fitted model."""

        if posterior:
            return tuple(
                [
                    self._sample(
                    self.idata.posterior.get(f'{self.__class__.__name__}::{var}').values.flatten(),
                     n_samples) 
                     for var in self._param_list
                    ]
                )

        else:
            return tuple(
                [
                    self.idata.posterior.get(f'{self.__class__.__name__}::{var}').mean().to_numpy()
                    for var in self._param_list
                    ]
                )

    def predict(self, 
        method:str,
        t: int = None, 
        n: int = None, 
        sample_posterior: bool =  False,
        posterior_draws: int = 100,
        rfm_df: pd.DataFrame = None,
        join_df = False 
        ) -> np.ndarray:
        """
        Predictive API.
        """

        if rfm_df is None:
            self._frequency, self._recency, self._T, self._monetary_value, _ = self._dataframe_parser(rfm_df)

        # TODO: Add exception handling for method argument.
        predictions = self._quantities_of_interest.get(method)(self,t,n,sample_posterior,posterior_draws)

        # TODO: Add arg to automatically merge to RFM dataframe?
        if join_df:
            pass
        
        if sample_posterior:
            # Additional columns will need to be added for mean, confidence intervals, etc.
            pass
        
        return predictions
        
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
    def _sample(param_array: array_like, n_samples: int) -> np.ndarray:
        """Utility function for sampling from parameter posteriors."""
        rng = np.random.default_rng()
        return rng.choice(param_array, n_samples, replace=True)


class PredictMixin(ABC, Generic[SELF]):
    """
    Define predictive methods for all models except GammaGamma.
    In research literature these are commonly referred to as quantities of interest.
    """

    @abstractmethod
    def _conditional_probability_alive(
        self, 
        t: float = None, 
        n: int = None, 
        sample_posterior: bool = False,
        posterior_draws: int = 100
        ) -> None:
        pass
    
    @abstractmethod
    def _conditional_expected_number_of_purchases_up_to_time(
        self, 
        t: float = None, 
        n: int = None, 
        sample_posterior: bool = False,
        posterior_draws: int = 100
        ) -> None:
        pass
    
    @abstractmethod
    def _expected_number_of_purchases_up_to_time(
        self, 
        t: float = None, 
        n: int = None, 
        sample_posterior: bool = False,
        posterior_draws: int = 100
        ) -> None:
        pass
    
    @abstractmethod
    def _probability_of_n_purchases_up_to_time(
        self, 
        t: float = None, 
        n: int = None, 
        sample_posterior: bool = False,
        posterior_draws: int = 100
        ) -> None:
        pass
