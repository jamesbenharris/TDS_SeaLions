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
    if (r>150 and g<80 and b<80):
        return True,'Red'
    elif (r<40 and g<70 and b>150):
        return True,'Blue'
    elif (r>70 and r<100 and g<60 and g>30 and b<20):
        return True,'Brown'
    elif (r<60 and g>150 and b<80):
        return True,'Green'    
    elif (r>220 and g<50 and b>220):
        return True,'Pink'
    else:
        return False,'Unknown'

def replaceColor(np.ndarray tempImage):
    cdef int x = tempImage.shape[1]   
    cdef int y = tempImage.shape[0]
    cdef int w, h
    for w in range(x):
        for h in range(y):
            truth, color = getColor(tempImage[h,w])
            if (truth):
                tempImage[h,w] = 0
    return tempImage

def getSquare(np.ndarray tempContour):
    cdef int x = min(tempContour[:, 1])-20
    cdef int y = min(tempContour[:, 0])-20
    cdef int w = (max(tempContour[:, 1])-min(tempContour[:, 1]))+20
    cdef int h = (max(tempContour[:, 0])-min(tempContour[:, 0]))+20
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
    
def getBlack(np.ndarray tempImage):
    cdef int r = tempImage[0]
    cdef int g = tempImage[1]
    cdef int b = tempImage[2]
    if(r==0 and g==0 and b==0):
        return True
    else:
        return False