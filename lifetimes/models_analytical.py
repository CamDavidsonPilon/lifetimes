
class ModelAnalytical(object):
    """
    Models defined just by parameters, with analytical available quantities.
    """
    # TODO: in the future might be extended by adding analytical errors

    def __init__(self, pars):
        """
        Args:
            pars: dictionary with input parameters
        """
        super(ModelAnalytical, self).__init__()
        self.params_ = pars

    def _print_params(self):
        s = ""
        for p, value in self.params_.items():
            s += "%s: %.2f, " % (p, value)
        return s.strip(', ')


    # TODO: wrap analytical calculations already contained in 'estimation.py' to make them live independently from a fit



