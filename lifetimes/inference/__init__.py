"""BaseInferencer class for all lifetimes inference objects.

This constructor object defines the BaseInferencer, containing base methods for inheritance by all Bayesian inference models.
It is intended to be used whenever defining classes for new models in the library:

Usage:
    Class <ModelName>Inferencer(BaseInferencer):

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

from __future__ import generator_stop
from __future__ import annotations

import warnings

import pymc as pm
import aesara.tensor as at

from ..utils import *


class BaseInference:
    """
    Base inference class for all inferencers.

    Attributes
    ----------
    attr_name : datatype
        Add description here.

    Methods
    -------
    summary(self)
        Represent the photo in the given colorspace.

    """

    def __repr__(self) -> str:
        """Representation of Inference model object."""
        classname = self.__class__.__name__
        try:
            row_str = f"Estimated with {self.data.shape[0]} samples."
            return f"<btyd.{classname}: {row_str}>"
        except AttributeError:
            return f"<btyd.{classname}>"

    @property
    def summary(self):
        """
        Summary statistics for posterior inference.

        Returns
        -------
        var_name : datatype
            Add description here.

        See Also
        --------
        ``..utils.plotting.py``
        """
        
        pass
