# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:54:04 2019

@author: sicatrb
"""

import cv2
import numpy as np
import sys
import math

filename = "l0.png"

image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

imagefloat = image.view((np.float32,1))

def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = math.floor(diameter/2)
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = i_filtered


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image

filtered_image = bilateral_filter_own(imagefloat, 5, 12.0, 16.0)
cv2.imwrite("filtered_image.png", filtered_image.view((np.uint8, 4)))

if __name__ == "__main__":
    filename = str(sys.argv[1])
    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED).view((np.float32, 1))
    
    #filtered_image_OpenCV = cv2.bilateralFilter(src, 5, 12.0, 16.0)
    #cv2.imwrite("original_image_grayscale.png", src)
    #cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
    diameter = sys.argv[2]
    sigma_i = sys.argv[3]
    sigma_space = sys.argv[4]
    filtered_image = bilateral_filter_own(src, diameter, sigma_i, sigma_space)
    cv2.imwrite("filtered_" + filename  + "_d" + diameter + "_si" + sigma_i +
                "_ss" + sigma_space + ".png", filtered_image)