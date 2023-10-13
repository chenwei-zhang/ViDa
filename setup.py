#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='vida',
      version='1.0.0',
      description='Visualizing DNA Kinetics',
      author='Chenwei Zhang',
      author_email='cwzhang@cs.ubc.ca',
      url='https://github.com/chenwei-zhang/ViDa',
      install_requires=[
            'torch'
      ],
      packages=find_packages()
      )
