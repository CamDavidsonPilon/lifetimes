## Saving and loading model

When you have lots of data and training takes a lot of time option with saving and loading model could be useful. First you need to fit the model, then save it and load.

### Fit model

```python
from lifetimes import BetaGeoFitter
from lifetimes.datasets import load_cdnow_summary

data = load_cdnow_summary(index_col=[0])
bgf = BetaGeoFitter()
bgf.fit(data['frequency'], data['recency'], data['T'])
bgf
"""<lifetimes.BetaGeoFitter: fitted with 2357 subjects, a: 0.79, alpha: 4.41, b: 2.43, r: 0.24>"""
```

### Saving model

Model will be saved with [dill](https://github.com/uqfoundation/dill) to pickle object. Optional parameters `save_data` and `save_generate_data_method` are present to reduce final pickle object size for big dataframes.
Optional parameters:
- `save_data` is used for saving data from model or not (default: `True`).
- `save_generate_data_method` is used for saving `generate_new_data` method from model or not (default: `True`)

```python
bgf.save_model('bgf.pkl')
```

or to save only model with minumum size without `data` and `generate_new_data`:
```python
bgf.save_model('bgf_small_size.pkl', save_data=False, save_generate_data_method=False)
```

### Loading model

Before loading you should initialize the model first and then use method `load_model`

```python
bgf_loaded = BetaGeoFitter()
bgf_loaded.load_model('bgf.pkl')
bgf_loaded
"""<lifetimes.BetaGeoFitter: fitted with 2357 subjects, a: 0.79, alpha: 4.41, b: 2.43, r: 0.24>"""
```
