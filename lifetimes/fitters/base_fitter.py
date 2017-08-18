"""Base fitter for other classes."""
import dill
from ..utils import _save_obj_without_attr


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
            param_str = ", ".join("{}: {:.2f}".format(par, val) for par, val
                                  in sorted(self.params_.items()))
            return "<lifetimes.{classname}:{subj_str} {param_str}>".format(
                classname=classname, subj_str=subj_str, param_str=param_str)
        except AttributeError:
            return "<lifetimes.{classname}>".format(classname=classname)

    def _unload_params(self, *args):
        if not hasattr(self, 'params_'):
            raise ValueError("Model has not been fit yet. Please call the .fit"
                             " method first.")
        return [self.params_[x] for x in args]

    def save_model(self, path, save_data=True, save_generate_data_method=True,
                   values_to_save=None):
        """
        Save model with dill package.

        Parameters
        ----------
        path: str
            Path where to save model.
        save_date: bool, optional
            Whether to save data from fitter.data to pickle object
        save_generate_data_method: bool, optional
            Whether to save generate_new_data method (if it exists) from
            fitter.generate_new_data to pickle object.
        values_to_save: list, optional
            Placeholders for original attributes for saving object. If None
            will be extended to attr_list length like [None] * len(attr_list)

        """
        attr_list = ['data' * (not save_data),
                     'generate_new_data' * (not save_generate_data_method)]
        _save_obj_without_attr(self, attr_list, path,
                               values_to_save=values_to_save)

    def load_model(self, path):
        """
        Load model with dill package.

        Parameters
        ----------
        path: str
            From what path load model.

        """
        with open(path, 'rb') as in_file:
            self.__dict__.update(dill.load(in_file).__dict__)
