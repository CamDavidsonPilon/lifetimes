from __future__ import print_function
import pytest
from lifetimes.formulas import gamma_ratio


@pytest.mark.BGBB
def test_gamma_ratio():

    xs =  [10.0,100.0,1000.0,10000.0,100000.0]
    gr = []

    for x in xs:
        gr.append(gamma_ratio(x, 1))

    print(gr)