# -*- coding: utf-8 -*-
# modified from https://github.com/CamDavidsonPilon/lifelines/

import pandas as pd
from pkg_resources import resource_filename


def load_dataset(filename, **kwargs):
    '''
    Load a dataset from lifetimes.datasets

    Parameters:
    filename : for example "larynx.csv"
    usecols : list of columns in file to use

    Returns : Pandas dataframe
    '''
    return pd.read_csv(resource_filename('lifetimes', 'datasets/' + filename), **kwargs)


def load_cdnow(**kwargs):
    return load_dataset('cdnow_customers.csv', **kwargs)


def load_transaction_data(**kwargs):
    return load_dataset('example_transactions.csv', **kwargs)
