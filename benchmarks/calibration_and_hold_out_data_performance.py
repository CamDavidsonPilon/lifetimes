"""
Script demonstrating performance speed up on calibration_and_hold_out_data()
Speed up is as compared with code in commit:  e9b3475ec81c3a036fd8087f50b65db70a651e75
"""
import timeit
from lifetimes.utils import calibration_and_holdout_data
from lifetimes.datasets import load_transaction_data


def run():
    code = (
        '''calibration_and_holdout_data(transaction_data, 'id', 'date',
                                        calibration_period_end='2014-09-01',
                                        observation_period_end='2014-12-31' )''')

    print('running: {}'.format(code))
    result = timeit.timeit(code, setup='transaction_data = load_transaction_data()', globals=globals(), number=1)
    return result


if __name__ == '__main__':
    print(run())
