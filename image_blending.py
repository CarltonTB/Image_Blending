# Author: Carlton Brady
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


def convolve(I, H):
    """
    I is an image of varying size
    H is a kernel of varying size
    both should be numpy arrays
    returns:
    A new image that is the result of the convolution
    """
    # Stride length is assumed to be 1
    # axis zero is number of rows
    # axis one is number of columns
    # filter height
    filter_height = np.size(H, 0)
    # filter width
    filter_width = np.size(H, 1)
    # image depth
    image_channels = np.size(I, 2)
    # image height
    image_height = np.size(I, 0)
    # image width
    image_width = np.size(I, 1)
    padding_width = int(math.floor(filter_width-1)/2)
    padding_height = int(math.floor(filter_height-1)/2)
    padded_image = apply_padding(I, padding_height, padding_width)
    image_height_with_padding = np.size(padded_image, 0)
    image_width_with_padding = np.size(padded_image, 1)
    output_width = image_width_with_padding - filter_width + 1
    output_height = image_height_with_padding - filter_height + 1
    # The padding should cause the output image to have the same dimensions as the input image
    assert(output_width == image_width)
    assert(output_height == image_height)
    result_list = []
    for i in range(0, output_height):
        new_row_list = []
        for j in range(0, output_width):
            new_channel_list = []
            for k in range(0, image_channels):
                image_slice = padded_image[i:i+filter_height, j:j+filter_width, k]
                dot_product = np.sum(H*image_slice)
                dot_product = int(round(dot_product, 0))
                new_channel_list.append(dot_product)
                # print(image_slice)
            new_row_list.append(new_channel_list)
        result_list.append(new_row_list)
    result = np.array(result_list)
    return result.astype(np.uint8)


def reduce(I):
    """
    I is an image of varying size
    Does gaussian blurring then samples every other pixel
    returns:
    a copy of the image down sampled to be half the height and half the width
    """
    row_gaussian_kernel = np.array([[1, 2, 1]],
                                   dtype=np.float32)
    row_gaussian_kernel = (1 / 4) * row_gaussian_kernel
    col_gaussian_kernel = row_gaussian_kernel.transpose()
    # Blur with the column filter first
    col_kernel_blurred_image = convolve(I, col_gaussian_kernel)
    # Blur with the row filter
    blurred_image = convolve(col_kernel_blurred_image, row_gaussian_kernel)
    # image height
    image_height = np.size(blurred_image, 0)
    # image width
    image_width = np.size(blurred_image, 1)
    result_list = []
    for i in range(0, image_height, 2):
        new_row_list = []
        for j in range(0, image_width, 2):
            new_row_list.append(blurred_image[i, j])
        result_list.append(new_row_list)
    result = np.array(result_list)
    return result.astype(np.uint8)


def expand(I):
    """
    I is an image of varying size
    returns:
    a copy of the image expanded to be twice the size
    """
    # image depth
    image_channels = np.size(I, 2)
    # image height
    image_height = np.size(I, 0)
    # image width
    image_width = np.size(I, 1)
    # create an ndarray of twice the size filled with all zeros, then fill it in
    result = np.zeros((image_height*2, image_width*2, image_channels), dtype=np.uint8)
    pr = 0  # expanded pixel row
    # loop through all the pixels in the image
    for i in range(0, image_height):
        pc = 0  # expanded pixel col
        for j in range(0, image_width):
            # make a 2x2 square of pixels in the output equal to the current pixel
            result[pr:pr+2, pc:pc+2] = I[i, j]
            pc += 2
        pr += 2
    return result


def gaussian_pyramid(I, n):
    """
    Creates a Gaussian pyramid of the image with n levels.
    I is an image of varying size
    n is the number of levels in the pyramid
    return:
    a list of images in the gaussian pyramid from largest to smallest.
    each image is a numpy ndarray.
    """
    g_pyramid = []
    cur_level = np.copy(I)
    g_pyramid.append(cur_level)
    for i in range(0, n):
        cur_level = reduce(cur_level)
        g_pyramid.append(cur_level)
    return g_pyramid


def laplacian_pyramid(I, n):
    """
    Creates a Laplacian pyramid for the image by taking the difference of Gaussians.
    I is an image of varying size
    n is the number of levels in the pyramid
    returns:
    a list of images in the laplacian pyramid from largest to smallest.
    each image is a numpy ndarray.
    """
    # first create the gaussian pyramid
    g_pyramid = gaussian_pyramid(I, n)
    l_pyramid = [None]*n
    # the smallest levels are the same in each pyramid
    l_pyramid[n-1] = g_pyramid[n-1]
    for i in range(0, n-1):
        expanded_image = expand(g_pyramid[i+1])
        desired_dimensions = np.shape(g_pyramid[i])
        # in case the dimensions are off by 1 from rounding
        if desired_dimensions != np.shape(expanded_image):
            expanded_image = cv2.resize(expanded_image, dsize=(desired_dimensions[1], desired_dimensions[0]))
        l_pyramid[i] = g_pyramid[i] - expanded_image
    return l_pyramid


def apply_padding(I, padding_height, padding_width):
    """
    Helper function that applies zero-padding
    I is an image of varying size
    padding_height is the thickness of the padding to be adding on top and bottom
    padding_width is the thickness of the padding to be added on left and right
    returns;
    A new image (numpy ndarray) with the added zero-padding
    """
    # image depth
    image_channels = np.size(I, 2)
    # image height
    image_height = np.size(I, 0)
    # image width
    image_width = np.size(I, 1)
    zero_row = np.array([[[0]*image_channels]*image_width])
    zero_column = np.array([[[0]*image_channels]]*(image_height+padding_height*2))
    result = np.copy(I)
    for i in range(0, padding_height):
        result = np.concatenate((zero_row, result), axis=0)
        result = np.concatenate((result, zero_row), axis=0)
    for j in range(0, padding_width):
        result = np.concatenate((zero_column, result), axis=1)
        result = np.concatenate((result, zero_column), axis=1)
    return result
