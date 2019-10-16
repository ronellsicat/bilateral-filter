# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:54:04 2019

@author: sicatrb
"""

import cv2
import numpy as np
import math

def gaussianF(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))

def gaussianApprox(x, sigma):
    
    alpha = 0.5 / sigma
    
    x *= alpha
    if(x < -2):
	    return 0
    if(x < -1):
	    x += 2
	    return x * x * x
    if(x < 0):
	    return (4 - 3*x*x*(2 + x))
    if(x < 1):
	    return (4 - 3*x*x*(2 - x))
    if(x < 2):
	    x = 2-x
	    return x * x * x
    return 0

def filter_pixel(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = math.floor(diameter/2)
    c = 0
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        neighbour_x = x - (hl - i)
        if neighbour_x >= len(source):
            neighbour_x -= len(source)
                
        gx = gaussianApprox(neighbour_x - x, sigma_s)
        
        while j < diameter:
            
            neighbour_y = y - (hl - j)   
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gy = gaussianApprox(neighbour_y - y, sigma_s)    
             
            gi = gaussianApprox(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gx * gy
            if(c == 0 and x == 10 and y == 10):       
                
                print("x,y = " + str(neighbour_x) + ", " + str(neighbour_y) + ": gi(" + str(source[neighbour_x][neighbour_y] - source[x][y]) + ") = " + str(gi) + ", gs = " + str(gs) + "\n")
            
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    c += 1
    i_filtered = i_filtered / Wp
    #print("in: ")
    #print(i_filtered)
    #print("out: ")
    filtered_image[x][y] = i_filtered
    #print(filtered_image[x][y])
    #if source[x][y] != i_filtered:
    #    print("not equal")


def bilateral_filter(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape, dtype=np.float32)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            filter_pixel(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image

def run_bilateral_filter(filename, diameter, sigma_i, sigma_space):
   
    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    src = src.view((np.float32, 1))
    
    print(src)
    
    i = 0
    while i < len(src):
        j = 0
        while j < len(src[0]):
            src[i][j] = src[i][j] + 1.0
            j += 1
        i += 1
    
    
    filtered_image = bilateral_filter(src, diameter, sigma_i, sigma_space)

    i = 0
    while i < len(src):
        j = 0
        while j < len(src[0]):
            filtered_image[i][j] = filtered_image[i][j] - 1.0
            j += 1
        i += 1

    
    print(filtered_image)
    
    filtered_image = filtered_image.view((np.uint8, 4))
    filtered_image = filtered_image[:,:,0,:]
    #print(filtered_image)
    
    cv2.imwrite("filtered_" + filename  + "_d" + str(diameter) + "_si" + str(sigma_i) +
                "_ss" + str(sigma_space) + ".png", filtered_image)
    
    return filtered_image