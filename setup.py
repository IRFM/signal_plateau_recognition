#!/usr/bin/env python

from setuptools import setup

setup(
    name="signal_plateau_recognition",
    version="0.0.1",
    description="Signal plateau recognition",
    author="Jorge Morales",
    author_email="jorge012@gmail.com",
    license="MIT",
    packages=["signal_plateau_recognition"],
    python_requires=">=3",
    install_requires=[
        "numpy<2",
        "scipy",
        "scikit-learn",
    ],
)
