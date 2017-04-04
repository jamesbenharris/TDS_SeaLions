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
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian,sobel
from skimage import data,io
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage import feature
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
from skimage import color
from webcolors import rgb_to_name
from skimage import novice
from search_image import getColor,replaceColor,getSquare,getCenter,getGreen,getBlack
    
# Convert to float: Important for subtraction later which won't work with uint8
orig = io.imread('/Users/benharris/Documents/Projects/SeaLions/images/0.jpg')
start_time = time.time()
orig2 = replaceColor(orig.copy())
print("--- %s seconds ---" % (time.time() - start_time))
gray = rgb2gray(orig2)
#image = io.imread('/Users/benharris/Documents/Projects/SeaLions/images/4.jpg',flatten=True)
#image = gaussian_filter(image, 1)

start_time = time.time()
edges3 = feature.canny(gray, sigma=1,low_threshold=.1,high_threshold=.15)
print("--- %s seconds ---" % (time.time() - start_time))
#edges3 = roberts(image)
io.imshow(edges3)
io.show()

start_time = time.time()
contours = find_contours(edges3, .7, fully_connected='high', positive_orientation='low')
print("--- %s seconds ---" % (time.time() - start_time))
fig, ax = plt.subplots()
ax.imshow(orig)

for n, contour in enumerate(contours):
    x_c,y_c = getCenter(contour)
    boolean = getBlack(orig2[y_c,x_c])
    #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    if(boolean):
        x,y,w,h = getSquare(contour)
        color = getColor(orig[y_c,x_c])[1]
        circ = plt.Circle((x_c, y_c), radius=.1, color='b')
        ax.add_patch(circ)
        ax.add_patch(
        patches.Rectangle(
            (x, y),
            h+20,
            w+20,
            fill=False      # remove background
        ))

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()