from estimation import BetaGeoFitter
import numpy as np
import pandas as pd
import generate_data as gen


class Model(object):
    def __init__(self, N, t):
        """

        Args:
            N:      number of users you're referring to
            t:      time horizon you're looking at
        Returns:

        """
        super(Model, self).__init__()

        if N < 100:
            raise ValueError("Number of users must be statistically relevant. At least 100.")
        if t < 0:
            raise ValueError("Time horizon must be positive.")

        self.t = t
        self.N = N
        self.fitted_model = None
        self.params, self.params_C = None, None
        self.numerical_metrics = None

    def fit(self, frequency, recency, T):
        """
        Fit the model to data, finding parameters and their errors, and assigning them to internal variables
        Args:
            frequency: the frequency vector of customers' purchases (denoted x in literature).
            recency: the recency vector of customers' purchases (denoted t_x in literature).
            T: the vector of customers' age (time since first purchase)
        """
        raise NotImplementedError()

    def simulate(self):
        """
        Runs a heavy simulation starting from parameters and their uncertainties,
        to calculate common metrics and their uncertainties.
        """
        raise NotImplementedError()


class BetaGeoModel(Model):
    """
    Fits a BetaGeoModel to the data, and computes relevant metrics by mean of a simulation.
    """

    def fit(self, frequency, recency, T):  # TODO: test it
        bgf = BetaGeoFitter()
        bgf.fit(frequency, recency, T)

        self.params = bgf.params_

        data = pd.DataFrame({'frequency': frequency, 'recency': recency, 'T': T})
        self.params_C = self.estimate_uncertainties_with_bootstrap(data)

        self.fitted_model = bgf

    @staticmethod
    def estimate_uncertainties_with_bootstrap(data, size=10):
        """
        Calculate parameter covariance Matrix by bootstrapping trainig data.

        Args:
            data:   pandas data farme containing 3 columns labelled 'frequency' 'recency' 'T'
            size:   nunmber of re-samplings

        Returns:    The estimated covariance matrix
        """

        if not all(column in data.columns for column in ['frequency', 'recency', 'T']):
            raise ValueError("given data do not contain the 3 magic columns.")
        if size < 2:
            raise ValueError("Run at least 2 samplings to get a covariance.")

        par_estimates = []

        for i in range(size):
            N = len(data)
            sampled_data = data.sample(N, replace=True)

            bgf = BetaGeoFitter(penalizer_coef=0.0)
            bgf.fit(sampled_data['frequency'], sampled_data['recency'], sampled_data['T'])
            par_estimates.append(bgf.params_)

        rs = [be['r'] for be in par_estimates]
        alphas = [be['alpha'] for be in par_estimates]
        As = [be['a'] for be in par_estimates]
        Bs = [be['b'] for be in par_estimates]

        np.cov(rs, alphas, As, Bs)
        x = np.vstack([rs, alphas, As, Bs])
        cov = np.cov(x)
        return cov

    def evaluate_metrics_with_uncertainties(self, N_syst=10, max_x=10):  # TODO: test it
        """
        Args:
            N_syst:        Number of re-sampling of parameters, for systematic errors
            max_x:         Maximum number of transactions you want to consider
        """

        if self.params is None or self.params_C is None:
            raise ValueError("Model has not been fit yet. Please call the .fit method first.")


        # extract probabilities together with systematic+statistical uncertainties (Montecarlo)
        par_samplings = np.random.multivariate_normal(self.params.values(), self.params_C, N_syst)

        xs = range(max_x)

        measurements_fx = {}
        for x in xs:
            measurements_fx[x] = []
        p_x = [None] * N_syst
        p_x_err = [None] * N_syst

        for par_s in par_samplings:
            r_s = par_s[0]
            alpha_s = par_s[1]
            a_s = par_s[2]
            b_s = par_s[3]

            data = gen.beta_geometric_nbd_model(self.t, r_s, alpha_s, a_s, b_s, size=self.N)
            n = len(data)
            for x in xs:
                if x == max_x - 1:
                    n_success = len(data[data['frequency'] >= x])  # the last bin is cumulative
                else:
                    n_success = len(data[data['frequency'] == x])
                measurements_fx[x].append(float(n_success) / n)

        for x in xs:
            p_x_err[x] = np.std(measurements_fx[x])  # contain statistic + systematic errors, entangled
            p_x[x] = np.mean(measurements_fx[x])

        self.numerical_metrics = NumericalMetrics(p_x, p_x_err)

    def simulate(self):
        self.evaluate_metrics_with_uncertainties()


class NumericalMetrics(object):
    """
    Contains the metrics common to all transaction counting models (Pareto/NBS, BG/NBD)
    """

    # TODO: add y, Ey, p to common metrics

    def __init__(self, p_x, p_x_err):
        """
        Args:
            p_x:    Probabilities of x (x being the number of transaction per user 0, 1, ...)
            p_x_err:    Error on probabilities of x (x being the number of transaction per user 0, 1, ...)
        """
        super(NumericalMetrics, self).__init__()
        self.p_x = p_x
        self.p_x_err = p_x_err

    def expected_x(self):
        """
        Returns:    The E[x] and error
        """
        # TODO: evaluate properly the error!
        raise NotImplementedError()
