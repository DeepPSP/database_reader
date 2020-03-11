# -*- coding: utf-8 -*-
from setuptools import setup
# from importlib.machinery import SourceFileLoader

from version import version as __version__

setup(name='database_reader',
      version=__version__,
      description='Reader for various databases',
      author='WEN Hao',
      author_email='wenh06@gmail.com',
      license='MIT',
    #   url='https://github.com/wenh06/database_reader',
      packages=['database_reader'],
      install_requires=[
        'numpy',
        'scipy',
        'pandas',
      ]
    )
