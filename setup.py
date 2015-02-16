#!/usr/bin/env python
import os
from distutils.core import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='Lifetimes',
      version='0.1.1',
      description='Measure customer lifetime value in Python',
      author='Cam Davidson-Pilon',
      author_email='cam.davidson.pilon@gmaillcom',
      packages=['lifetimes', 'lifetimes.datasets'],
      license="MIT",
      keywords="customer lifetime value, clv, ltv, BG/NBD, pareto/NBD",
      url="https://github.com/CamDavidsonPilon/lifetimes",
      long_description=read('README.md'),
      classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        ],
      install_requires=[
        "numpy",
        "scipy",
        "pandas>=0.14",
        ],
      package_data={
        "lifetimes": ["datasets/*",]
      }
     )
