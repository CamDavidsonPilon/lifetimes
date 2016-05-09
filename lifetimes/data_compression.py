import pandas as pd


def compress_data(data):
    """
    Takes id-level data and compress them by recency/frequency, data must contain columns 'frequency', 'recency', 'T'
    :param data:    id-level data
    :type data:     pandas df
    :return:        Compressed data frame
    """
    if 'T' not in data or 'frequency' not in data or 'recency' not in data:
        raise ValueError("Input data frame must contain recency, frequency and T")

    newData = []
    for T in pd.Series.unique(data['T']):
        data_T = data[data['T'] == T]
        for frequency in pd.Series.unique(data_T['frequency']):
            data_f = data_T[data_T['frequency'] == frequency]
            for recency in pd.Series.unique(data_f['recency']):
                N = len(data_f[data_f['recency'] == recency])
                newData.append([recency, frequency, T, N])

    compressed_data = pd.DataFrame(newData, columns=['recency', 'frequency', 'T', 'N'])
    return compressed_data


def filter_data_by_T(data, T1, T2):
    """
    Allows to quickly filter a data frame over T
    :return:    filtered data frame
    """
    if 'T' not in data:
        raise ValueError("Input data frame must contain T")

    return data[(data['T'] >= T1) & (data['T'] <= T2)]


