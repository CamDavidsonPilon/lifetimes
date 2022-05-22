"""All legacy MLE models from fitters directory and Bayesian models from inference directory.

License:
    MIT License (MIT) 

    Copyright (c) 2022, Colt Allen

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    1. The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from .fitters import BaseFitter
from .fitters.beta_geo_fitter import BetaGeoFitter
from .fitters.beta_geo_beta_binom_fitter import BetaGeoBetaBinomFitter
from .fitters.modified_beta_geo_fitter import ModifiedBetaGeoFitter
from .fitters.pareto_nbd_fitter import ParetoNBDFitter
from .fitters.gamma_gamma_fitter import GammaGammaFitter
from .fitters.beta_geo_covar_fitter import BetaGeoCovarsFitter
from .models import BaseInferencer

__version__ = "0.1.0"

__all__ = (
    "__version__",
    "BetaGeoFitter",
    "ParetoNBDFitter",
    "GammaGammaFitter",
    "ModifiedBetaGeoFitter",
    "BetaGeoBetaBinomFitter",
    "BetaGeoCovarsFitter"
)
