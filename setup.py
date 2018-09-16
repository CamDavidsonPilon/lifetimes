#!/usr/bin/env python
import os
from setuptools import setup


exec(compile(open('lifetimes/version.py').read(),
             'lifetimes/version.py', 'exec'))


readme_path = os.path.join(os.path.dirname(__file__), 'README.md')

try:
    import pypandoc
    long_description = pypandoc.convert_file(readme_path, 'rst')
except(ImportError):
    long_description = open(readme_path).read()


setup(name='Lifetimes',
      version=__version__,
      description='Measure customer lifetime value in Python',
      author='Cam Davidson-Pilon',
      author_email='cam.davidson.pilon@gmail.com',
      packages=['lifetimes', 'lifetimes.datasets'],
      license="MIT",
      keywords="customer lifetime value, clv, ltv, BG/NBD, pareto/NBD, frequency, recency",
      url="https://github.com/CamDavidsonPilon/lifetimes",
      long_description=long_description,
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
          "dill"
      ],
      package_data={
          "lifetimes": [
              "datasets/*",
              "../README.md",
              "../README.txt",
              "../LICENSE",
              "../MANIFEST.in",
              "fitters/*"
          ]
      }
      )
