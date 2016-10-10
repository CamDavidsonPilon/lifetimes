#!/usr/bin/env python
import os
from distutils.core import setup

exec(compile(open('lifetimes/version.py').read(),
                  'lifetimes/version.py', 'exec'))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='Lifetimes',
      version=__version__,
      description='Measure customer lifetime value in Python',
      author='Cam Davidson-Pilon',
      author_email='cam.davidson.pilon@gmaillcom',
      packages=['lifetimes', 'lifetimes.datasets'],
      license="MIT",
      keywords="customer lifetime value, clv, ltv, BG/NBD, pareto/NBD, frequency, recency",
      url="https://github.com/CamDavidsonPilon/lifetimes",
      long_description=read('README.md'),
      classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering",
        ],
      install_requires=[
        "numpy",
        "scipy",
        "pandas>=0.19",
        ],
      package_data={
        "lifetimes": [
                    "datasets/*",
                    "../README.md",
                    "../README.txt",
                    "../LICENSE",
                    "../MANIFEST.in",
          ]
      }
     )
