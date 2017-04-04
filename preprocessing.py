#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:31:04 2017

@author: benharris
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from skimage.color import rgb2gray,rgba2rgb
from skimage import data,io
from skimage import feature
from skimage.measure import find_contours
from skimage import color
from search_image import getColor,replaceColor,getSquare,getCenter,getGreen,getBool

#Read in image, replace colors with black then converts to gray. Allows find edges to work better.
def processImg(img):
    orig = io.imread(img)
    start_time = time.time()
    orig2 = replaceColor(orig.copy())
    print("--- Cleaned image in %s seconds ---" % (time.time() - start_time))
    gray = rgb2gray(orig2)
    
    #Find Edges
    start_time = time.time()
    edges3 = feature.canny(gray, sigma=3,low_threshold=.1,high_threshold=.15)
    print("--- Found edges in %s seconds ---" % (time.time() - start_time))
    #edges3 = roberts(image)
    #io.imshow(edges3)
    #io.show()
    
    #Find Contours
    start_time = time.time()
    contours = find_contours(edges3, .8, fully_connected='high', positive_orientation='low')
    print("--- Found contours in %s seconds ---" % (time.time() - start_time))
    return contours,orig,orig2,edges3

def extractSeaLions(contours):
    start_time = time.time()
    #Filter contours by color and shape. Plot Image with squares
    for n, contour in enumerate(contours):
        x_c,y_c = getCenter(contour)
        x,y,w,h = getSquare(contour)
        boolean = getBool(orig2[y_c,x_c])
        #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        if(boolean and abs(w-h)<10):
            color = getColor(orig[y_c,x_c])[1]
                        
    print("--- Extracted Sea Lions in %s seconds ---" % (time.time() - start_time))

def findSeaLions(contours,orig,orig2,shape):
    start_time = time.time()
    fig, ax = plt.subplots()
    ax.imshow(orig)
    #Filter contours by color and shape. Plot Image with squares
    for n, contour in enumerate(contours):
        x_c,y_c = getCenter(contour)
        boolean = getBool(orig2[y_c,x_c])
        truth,cl,scale = getColor(orig[y_c,x_c])
        x,y,w,h = getSquare(contour,scale)
        #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        if(boolean and abs(w-h)<5):
            circ = plt.Circle((x_c, y_c), radius=.1, color='b')
            ax.add_patch(circ)
            ax.add_patch(
            patches.Rectangle(
                (x, y),
                h,
                w,
                fill=False      # remove background
            ))  
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    print("--- Found Sea Lions in %s seconds ---" % (time.time() - start_time))

