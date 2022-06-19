# BTYD

[![Actively Maintained](https://img.shields.io/badge/Development%20Status-Active%20-yellowgreen.svg)](https://gist.github.com/cheerfulstoic/d107229326a01ff0f333a1d3476e068d)
[![PyPI version](https://badge.fury.io/py/btyd.svg)](https://badge.fury.io/py/Lifetimes)
[![GitHub license](https://img.shields.io/github/license/ColtAllen/btyd)](https://github.com/ColtAllen/btyd/blob/master/LICENSE.txt)


## READ FIRST: Project Status

BTYD is the successor to the [Lifetimes](https://github.com/CamDavidsonPilon/lifetimes) library for implementing Buy Till You Die and Customer Lifetime Value statistical models in Python. All existing Lifetimes functionality is supported, and Bayesian [PyMC](https://github.com/pymc-devs) model implementations are presently in an alpha stage of development.


## Introduction

BTYD can be used to analyze your users based on the following assumptions:

1. Users interact with you when they are active, or "alive"
2. Users under study may "die" or become inactive after some period of time

If this is too abstract, consider these applications:

 - Predicting how often a visitor will return to your website. (Alive = visiting. Die = decided the website wasn't for them)
 - Understanding how frequently a patient may return to a hospital. (Alive = visiting. Die = maybe the patient moved to a new city, or became deceased.)
 - Predicting individuals who have churned from an app using only their usage history. (Alive = logins. Die = removed the app)
 - Predicting repeat purchases from a customer. (Alive = actively purchasing. Die = became disinterested with your product)
 - Predicting the lifetime value of your customers


## Installation

```bash
pip install btyd
```

## Contributing

Please refer to the [Contributing Guide](https://github.com/ColtAllen/btyd/blob/master/CONTRIBUTING.md) before creating any *Pull Requests*.

## Documentation and tutorials
[Official Documentation of pre-fork Lifetimes library](http://lifetimes.readthedocs.io/en/latest/)


## Questions? Comments? Requests?

Please create an issue in the [BTYD repository](https://github.com/ColtAllen/btyd).

## Supported Models

- **BG/NBD** Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee (2005a),
       ["Counting Your Customers the Easy Way: An Alternative to the
       Pareto/NBD Model"](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf), Marketing Science, 24 (2), 275-84.

## Additional Information

1. R implementation is called [BTYDplus](https://github.com/mplatzer/BTYDplus).
1. [Bruce Hardie's website](http://brucehardie.com/), especially his notes, is full of useful and essential explanations, many of which are featured in this library.
