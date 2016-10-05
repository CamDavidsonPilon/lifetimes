import math
import pytest
from lifetimes.generate_data import generate_monetary_values, sample_monetary_values
import numpy as np


@pytest.mark.generate
def test_generate_monetary_value():
    values = [1, 10, 100]
    probs = [8, 1, 1]

    sampled_values = generate_monetary_values(values, probs, 1)

    assert len(sampled_values) == 1
    assert sampled_values[0] in values

    sampled_values = generate_monetary_values(values, probs, 1000)

    assert len(sampled_values) == 1000
    frequencies = [
        sum([1 for v in sampled_values if v == 1]),
        sum([1 for v in sampled_values if v == 10]),
        sum([1 for v in sampled_values if v == 100]),
    ]
    frequencies = [float(f) / 1000 for f in frequencies]
    assert math.fabs(frequencies[0] - 0.8) < 0.1
    assert math.fabs(frequencies[1] - 0.1) < 0.05
    assert math.fabs(frequencies[2] - 0.1) < 0.05


@pytest.mark.generate
def test_sample_monetary_value_from_csv():
    sampled_values = sample_monetary_values(size=100)

    assert len(sampled_values) == 100
    assert math.fabs(np.mean(sampled_values) - 8) < 3
    assert math.fabs(np.std(sampled_values) - 20) < 20
