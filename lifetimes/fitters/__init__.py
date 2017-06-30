"""Init for fitters."""

import dill


class BaseFitter(object):
    """Base class for fitters."""

    def __repr__(self):
        """Representation of fitter."""
        classname = self.__class__.__name__
        try:
            param_str = ", ".join("%s: %.2f" % (param, value) for param, value
                                  in sorted(self.params_.items()))
            return "<lifetimes.%s: fitted with %d subjects, %s>" % (
                classname, self.data.shape[0], param_str)
        except AttributeError:
            return "<lifetimes.%s>" % classname

    def _unload_params(self, *args):
        if not hasattr(self, 'params_'):
            raise ValueError("Model has not been fit yet. Please call the .fit"
                             " method first.")
        return [self.params_[x] for x in args]

    def save_model(self, path, save_data=True):
        """Save model with dill package.

        Parameters:
            path: Path where to save model.
            save_date: Whether to save data from fitter.data to pickle object

        """
        with open(path, 'wb') as out_file:
            if save_data:
                dill.dump(self, out_file)
            else:
                self_data = self.data.copy()
                self.data = []
                dill.dump(self, out_file)
                self.data = self_data

    def load_model(self, path):
        """Save model with dill package.

        Parameters:
            path: From what path load model.

        """
        with open(path, 'rb') as in_file:
            self.__dict__.update(dill.load(in_file).__dict__)
