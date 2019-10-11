#!/usr/bin/env python

from distutils.core import setup

setup(name='signal_plateau_recognition',
      version='0.1',
      description='Signal plateau recognition',
      author='Jorge Morales',
      author_email='jorge012@gmail.com',
      packages=['signal_plateau_recognition'],
      install_requires = ['numpy>=1.15', 'scipy>=1.0', 'sklearn>=0.18'],
     )
