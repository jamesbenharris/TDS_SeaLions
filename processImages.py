#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 00:47:51 2017

@author: benharris
"""

from preprocessing import *

results,image,image_cleaned  = processImg('/Users/benharris/Documents/Projects/SeaLions/images/0.jpg')

findSeaLions(results,image,image_cleaned)