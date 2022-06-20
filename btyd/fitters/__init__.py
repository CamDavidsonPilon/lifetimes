# -*- coding: utf-8 -*-
"""Base fitter for other classes."""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import dill
import numpy as np
import pandas as pd
from textwrap import dedent
from scipy.optimize import minimize
from autograd import value_and_grad, hessian
from ..utils import _save_obj_without_attr, ConvergenceError


class BaseFitter(object):
    """Base class for fitters."""

    def __repr__(self):
        """Representation of fitter."""
        classname = self.__class__.__name__
        try:
            subj_str = " fitted with {:d} subjects,".format(self.data.shape[0])
        except AttributeError:
            subj_str = ""

        try:
            param_str = ", ".join("{}: {}".format(par, val) for par, val in sorted(self.params_.items()))
            return "<btyd.{classname}:{subj_str} {param_str}>".format(
                classname=classname, subj_str=subj_str, param_str=param_str
            )
        except AttributeError:
            return "<btyd.{classname}>".format(classname=classname)

    def _unload_params(self, *args):
        if not hasattr(self, "params_"):
            raise ValueError("Model has not been fit yet. Please call the .fit" " method first.")
        return [self.params_[x] for x in args]

    def save_model(self, path, save_data=True, save_generate_data_method=True, values_to_save=None):
        """
        Save model with dill package.

        Parameters
        ----------
        path: str
            Path where to save model.
        save_data: bool, optional
            Whether to save data from fitter.data to pickle object
        save_generate_data_method: bool, optional
            Whether to save generate_new_data method (if it exists) from
            fitter.generate_new_data to pickle object.
        values_to_save: list, optional
            Placeholders for original attributes for saving object. If None
            will be extended to attr_list length like [None] * len(attr_list)

        """
        attr_list = ["data" * (not save_data), "generate_new_data" * (not save_generate_data_method)]
        _save_obj_without_attr(self, attr_list, path, values_to_save=values_to_save)

    def load_model(self, path):
        """
        Load model with dill package.

        Parameters
        ----------
        path: str
            From what path load model.

        """
        with open(path, "rb") as in_file:
            self.__dict__.update(dill.load(in_file).__dict__)

    def _compute_variance_matrix(self):
        params_ = self.params_
        return pd.DataFrame(
            (params_ ** 2).values * np.linalg.inv(self._hessian_) / self.data["weights"].sum(),
            columns=params_.index,
            index=params_.index,
        )

    def _compute_standard_errors(self):
        return np.sqrt(pd.Series(np.diag(self.variance_matrix_.values), index=self.params_.index))

    def _compute_confidence_intervals(self):
        inv_cdf_at_5_confidence = 1.96
        return pd.DataFrame(
            {
                "lower 95% bound": self.params_ - inv_cdf_at_5_confidence * self.standard_errors_,
                "upper 95% bound": self.params_ + inv_cdf_at_5_confidence * self.standard_errors_,
            },
            index=self.params_.index,
        )

    def _fit(self, minimizing_function_args, initial_params, params_size, disp, tol=1e-7, bounds=None, **kwargs):
        # set options for minimize, if specified in kwargs will be overwritten
        minimize_options = {}
        minimize_options["disp"] = disp
        minimize_options.update(kwargs)

        current_init_params = 0.1 * np.ones(params_size) if initial_params is None else initial_params
        output = minimize(
            value_and_grad(self._negative_log_likelihood),
            jac=True,
            method=None,
            tol=tol,
            x0=current_init_params,
            args=minimizing_function_args,
            options=minimize_options,
            bounds=bounds,
        )
        if output.success:
            hessian_ = hessian(self._negative_log_likelihood)(output.x, *minimizing_function_args)
            return output.x, output.fun, hessian_
        print(output)
        raise ConvergenceError(
            dedent(
                """
            The model did not converge. Try adding a larger penalizer to see if that helps convergence.
            """
            )
        )

    @property
    def summary(self):
        """
        Summary statistics describing the fit.

        Returns
        -------
        df : pd.DataFrame
            Contains columns coef, se(coef), lower, upper

        See Also
        --------
        ``print_summary``
        """
        df = pd.DataFrame(index=self.params_.index)
        df["coef"] = self.params_
        df["se(coef)"] = self.standard_errors_
        df["lower 95% bound"] = self.confidence_intervals_["lower 95% bound"]
        df["upper 95% bound"] = self.confidence_intervals_["upper 95% bound"]
        return df
