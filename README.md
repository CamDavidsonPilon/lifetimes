lifetimes
======================

Measuring customer lifetime value is hard. *lifetimes* makes it easy. 

## Quickstart
    
### Example using transactional data:

    from matplotlib import pyplot as plt
    from lifetimes.estimation import BetaGeoFitter
    from lifetimes.utils import summary_data_from_transaction_data

    # transactions representing customers' shopping history.
    transactions_A = pd.DataFrame([
        {'id': 1, 'date': '2015-01-01'},
        {'id': 1, 'date': '2015-01-05'},
        {'id': 1, 'date': '2015-01-08'},
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
        {'id': 3, 'date': '2015-02-07'},
    ])
    data_B = summary_data_from_transaction_data(transactions_B, 'id', 'date')


    bgf = BetaGeoFitter()
    bgf.fit(data_A['frequency'], data_A['recency'], data_A['cohort'])
    ax = bgf.plot(label='group A')

    bgf.fit(data_B['frequency'], data_B['recency'], data_B['cohort'])
    ax = bgf.plot(label='group B')
    plt.legend(loc='lower right')


![comp](http://i.imgur.com/ME70Qxgl.png)


### Example to assess model fit

    %matplotlib
    import pandas as pd
    from lifetimes.estimation import BetaGeoFitter

    df = pd.read_csv('lifetimes/datasets/cdnow_customers.csv', sep='\s+', index_col=[0])
    bg = BetaGeoFitter()
    bg.fit(df['x'], df['t_x'], df['T'] )

    bg.plot_period_transactions()

![model_fit](http://imgur.com/YDAdDrJl.png)




