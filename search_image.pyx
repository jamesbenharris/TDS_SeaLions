#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:26:14 2017

@author: benharris
"""
import numpy as np
cimport numpy as np

def getColor(np.ndarray colors):
    cdef int r = colors[0]
    cdef int g = colors[1]
    cdef int b = colors[2]
    if (r>71 and r<110 and g<65 and g>20 and b<30):
        return True,2,2
    elif (r>150 and g<80 and b<80):
        return True,0,3
    elif (r<40 and g<70 and b>130):
        return True,3,2
    elif (r<60 and g>150 and b<80):
        return True,4,.75
    elif (r>220 and g<50 and b>220):
        return True,1,3
    else:
        return False,'Unknown',0

def replaceColor(np.ndarray tempImage):
    cdef int x = tempImage.shape[1]   
    cdef int y = tempImage.shape[0]
    cdef int w, h
    for w in range(x):
        for h in range(y):
            truth = getColor(tempImage[h,w])[0]
            if (truth):
                tempImage[h,w] = [255,255,255]
            else:
                tempImage[h,w] = 0
    return tempImage

def getSquare(np.ndarray tempContour,scale):
    cdef int w = 45*scale
    cdef int h = 45*scale
    cdef int x = np.median(tempContour[:, 1])-w/2
    cdef int y = np.median(tempContour[:, 0])-h/2
    return x,y,h,w

def getCenter(np.ndarray tempContour):
    cdef int x = np.median(tempContour[:, 1])
    cdef int y = np.median(tempContour[:, 0])
    return (x,y)

def getGreen(np.ndarray tempImage):
    cdef int r = tempImage[0]
    cdef int g = tempImage[1]
    cdef int b = tempImage[2]
    if (r<60 and g>160 and b<50):
        return True
    else:
        return False
    
def getBool(np.ndarray tempImage):
    cdef int r = tempImage[0]
    cdef int g = tempImage[1]
    cdef int b = tempImage[2]
    if(r==255 and g==255 and b==255):
        return True
    else:
        return False