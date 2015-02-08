# Lifetimes

Measuring customer lifetime value is hard. Lifetimes makes it easy. 

## Quickstart

    from lifetimes.estimation import BetaGeoFitter
    from lifetimes.utils import summary_data_from_transaction_data

    transactions_A = pd.DataFrame([
        {'id': 1, 'date': '2015-01-01'},
        {'id': 1, 'date': '2015-01-05'},
        {'id': 2, 'date': '2015-01-04'},
        {'id': 2, 'date': '2015-01-30'},
        {'id': 3, 'date': '2015-01-10'},
        {'id': 3, 'date': '2015-02-05'},
        {'id': 3, 'date': '2015-02-06'},
    ])
    data_A = summary_data_from_transaction_data(transactions_A, 'id', 'date')

    transactions_B = pd.DataFrame([
        {'id': 1, 'date': '2015-01-01'},
        {'id': 1, 'date': '2015-02-05'},
        {'id': 2, 'date': '2015-02-01'},
        {'id': 2, 'date': '2015-02-05'},
        {'id': 3, 'date': '2015-01-16'},
        {'id': 3, 'date': '2015-02-05'},
        {'id': 3, 'date': '2015-02-06'},
    ])
    data_B = summary_data_from_transaction_data(transactions_B, 'id', 'date')


    bgf = BetaGeoFitter()
    bgf.fit(data_A['frequency'], data_A['recency'], data_A['cohort'])
    ax = bgf.plot(label='group A')

    bgf.fit(data_B['frequency'], data_B['recency'], data_B['cohort'])
    ax = bgf.plot(label='group B')


![comp](http://i.imgur.com/KK2zop0l.png)
