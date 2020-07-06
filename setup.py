#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
from setuptools import setup


exec(compile(open("lifetimes/version.py").read(), "lifetimes/version.py", "exec"))


readme_path = os.path.join(os.path.dirname(__file__), "README.md")


long_description = io.open(readme_path, encoding="utf8").read()


setup(
    name="Lifetimes",
    version=__version__,
    description="Measure customer lifetime value in Python",
    author="Cam Davidson-Pilon",
    author_email="cam.davidson.pilon@gmail.com",
    packages=["lifetimes", "lifetimes.datasets"],
    license="MIT",
    keywords="customer lifetime value, clv, ltv, BG/NBD, pareto/NBD, frequency, recency",
    url="https://github.com/CamDavidsonPilon/lifetimes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=["numpy>=1.10.0", "scipy>=1.0.0", "pandas>=0.24.0", "autograd>=1.2.0", "dill>=0.2.6"],
    package_data={
        "lifetimes": ["datasets/*", "../README.md", "../README.txt", "../LICENSE", "../MANIFEST.in", "fitters/*"]
    },
)
