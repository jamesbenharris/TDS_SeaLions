#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:33:31 2017

@author: benharris
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

 
setup(name="search_image", ext_modules=cythonize('search_image.pyx'),include_dirs=[numpy.get_include()])