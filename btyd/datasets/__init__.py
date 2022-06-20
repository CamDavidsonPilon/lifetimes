# -*- coding: utf-8 -*-
# modified from https://github.com/CamDavidsonPilon/lifelines/

import pandas as pd
from .. import utils
from pkg_resources import resource_filename

__all__ = [
    "load_cdnow_summary",
    "load_transaction_data",
    "load_cdnow_summary_data_with_monetary_value",
    "load_donations",
]


def load_dataset(filename, **kwargs):
    """
    Load a dataset from btyd.datasets.

    Parameters
    ----------
    filename: str
        for example "larynx.csv"
    usecols: list
        Passed to **kwargs, list of columns in file to use.
    **kwargs
        Passed to pandas.read_csv function.

    Returns
    -------
    DataFrame

    """
    return pd.read_csv(resource_filename("btyd", "datasets/" + filename), **kwargs)


def load_donations(**kwargs):
    """Load donations dataset as pandas DataFrame."""
    return load_dataset("donations.csv", **kwargs)


def load_cdnow_summary(**kwargs):
    """Load cdnow customers summary pandas DataFrame."""
    return load_dataset("cdnow_customers_summary.csv", **kwargs)


def load_transaction_data(**kwargs):
    """
    Return a Pandas dataframe of transactional data.

    Looks like:

                      date  id
    0  2014-03-08 00:00:00   0
    1  2014-05-21 00:00:00   1
    2  2014-03-14 00:00:00   2
    3  2014-04-09 00:00:00   2
    4  2014-05-21 00:00:00   2

    The data was artificially created using BTYD data generation routines. Data was generated
    between 2014-01-01 to 2014-12-31.

    """
    return load_dataset("example_transactions.csv", **kwargs)


def load_cdnow_summary_data_with_monetary_value(**kwargs):
    """Load cdnow customers summary with monetary value as pandas DataFrame."""
    df = load_dataset("cdnow_customers_summary_with_transactions.csv", **kwargs)
    df.columns = ["customer_id", "frequency", "recency", "T", "monetary_value"]
    df = df.set_index("customer_id")
    return df
