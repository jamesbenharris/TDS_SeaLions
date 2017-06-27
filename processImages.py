#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 00:47:51 2017d

@author: benharris
"""

from preprocessing import *

results,image,image_cleaned,edges  = processImg('/Users/benharris/Documents/Projects/SeaLions/images/1.jpg')

io.imshow(edges)
io.show()

findSeaLions(results,image,image_cleaned)
    
#ls = getTrainingList("/Volumes/Research/SeaLions/Kaggle-NOAA-SeaLions/TrainDotted")

#ls.to_csv("/Users/benharris/Documents/Projects/SeaLions/Results/SeaLions-R-CNN.csv", sep=',')

#images = extractSeaLions(results,image,image_cleaned)

#cleaned = cleanCSV("/Users/benharris/Documents/Projects/SeaLions/Results/SeaLions-R-CNN.csv","/Users/benharris/Documents/Projects/SeaLions/TrainSmall/Train/train.csv")
#cleaned.to_csv("/Users/benharris/Documents/Projects/SeaLions/Results/SeaLions-R-CNN-Cleaned.csv", sep=',')