![](http://i.imgur.com/7s3jqZM.png)

#### Measuring users is hard. Lifetimes makes it easy.
[![PyPI version](https://badge.fury.io/py/Lifetimes.svg)](https://badge.fury.io/py/Lifetimes)
[![Documentation Status](https://readthedocs.org/projects/lifetimes/badge/?version=latest)](http://lifetimes.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/CamDavidsonPilon/lifetimes.svg?branch=master)](https://travis-ci.org/CamDavidsonPilon/lifetimes)
[![Coverage Status](https://coveralls.io/repos/CamDavidsonPilon/lifetimes/badge.svg?branch=master)](https://coveralls.io/r/CamDavidsonPilon/lifetimes?branch=master)


## Introduction

Lifetimes can be used to analyze your users based on a few assumption:

1. Users interact with you when they are "alive".
2. Users under study may "die" after some period of time.

I've quoted "alive" and "die" as these are the most abstract terms: feel free to use your own definition of "alive" and "die" (they are used similarly to "birth" and "death" in survival analysis). Whenever we have individuals repeating occurrences, we can use Lifetimes to help understand user behaviour.

### Applications

If this is too abstract, consider these applications:

 - Predicting how often a visitor will return to your website. (Alive = visiting. Die = decided the website wasn't for them)
 - Understanding how frequently a patient may return to a hospital. (Alive = visiting. Die = maybe the patient moved to a new city, or became deceased.)
 - Predicting individuals who have churned from an app using only their usage history. (Alive = logins. Die = removed the app)
 - Predicting repeat purchases from a customer. (Alive = actively purchasing. Die = became disinterested with your product)
 - Predicting the lifetime value of your customers

### Specific Application: Customer Lifetime Value
As emphasized by P. Fader and B. Hardie, understanding and acting on customer lifetime value (CLV) is the most important part of your business's sales efforts. [And (apparently) everyone is doing it wrong](https://www.youtube.com/watch?v=guj2gVEEx4s). *Lifetimes* is a Python library to calculate CLV for you.


## Installation

    pip install lifetimes

Requirements are only Numpy, Scipy, Pandas, [Dill](https://github.com/uqfoundation/dill) (and optionally-but-seriously matplotlib).

## Documentation and tutorials
[Official documentation](http://lifetimes.readthedocs.io/en/latest/)


## Questions? Comments? Requests?

Please create an issue in the [lifetimes repository](https://github.com/CamDavidsonPilon/lifetimes). 


## More Information

1. [Roberto Medri](http://cdn.oreillystatic.com/en/assets/1/event/85/Case%20Study_%20What_s%20a%20Customer%20Worth_%20Presentation.pdf) did a nice presentation on CLV at Etsy.
2. [Papers](http://mktg.uni-svishtov.bg/ivm/resources/Counting_Your_Customers.pdf), lots of [papers](http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf).
3. R implementation is called [BTYD](http://cran.r-project.org/web/packages/BTYD/vignettes/BTYD-walkthrough.pdf) (for, *Buy 'Til You Die*).
