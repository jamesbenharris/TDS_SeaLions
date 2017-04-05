#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 00:47:51 2017d

@author: benharris
"""

from preprocessing import *

results,image,image_cleaned,edges  = processImg('/Users/benharris/Documents/Projects/SeaLions/images/0.jpg')

findSeaLions(results,image,image_cleaned)

images = extractSeaLions(results,image,image_cleaned)

io.imshow(edges)
io.show()
        
for img in images[:]:
    if len(img[0])>0 and img[1]==4:
        io.imshow(img[0])
        io.show()
        break

