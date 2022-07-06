import pytest

from btyd.datasets import load_cdnow_summary_data_with_monetary_value

import pandas as pd


@pytest.fixture(scope='module')
def cdnow_customers() -> pd.DataFrame:
    """ Create an RFM dataframe for multiple tests and fixtures. """
    rfm_df = load_cdnow_summary_data_with_monetary_value()
    return rfm_df