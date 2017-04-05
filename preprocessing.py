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
import pandas as pd
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
    edges3 = feature.canny(gray, sigma=4,low_threshold=.1,high_threshold=.15)
    print("--- Found edges in %s seconds ---" % (time.time() - start_time))
    #edges3 = roberts(image)
    #io.imshow(edges3)
    #io.show()
    
    #Find Contours
    start_time = time.time()
    contours = find_contours(edges3, .5, fully_connected='high', positive_orientation='low')
    print("--- Found contours in %s seconds ---" % (time.time() - start_time))
    return contours,orig,orig2,edges3

def extractSeaLions(contours,orig,orig2):
    start_time = time.time()
    df = []
    i = 0
    f = 0
    #Filter contours by color and shape. Plot Image with squares
    for n, contour in enumerate(contours):
        x_c,y_c = getCenter(contour)
        boolean = getBool(orig2[y_c,x_c])
        truth,cl,scale = getColor(orig[y_c,x_c])
        x,y,w,h = getSquare(contour,scale)
        i = (y,y+h,x,x+w)
        #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        if(boolean and abs(w-h)<10 and i != f):
            crop = orig[y:y+h,x:x+w]
            row = [crop,cl]
            df.append(row)
            f = (y,y+h,x,x+w)
    print("--- Extracted Sea Lions in %s seconds ---" % (time.time() - start_time))
    return df


def findSeaLions(contours,orig,orig2):
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
            circ = plt.Circle((x, y), radius=3, color='b')
            circ2 = plt.Circle((x+w, y+h), radius=3, color='r')
            ax.add_patch(circ)
            ax.add_patch(circ2)
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

