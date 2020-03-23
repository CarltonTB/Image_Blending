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
    # TODO: add padding so the result is of the same dimensions as the original image
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
    return None


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
    # print(zero_row)
    # print(zero_column)
    result = np.copy(I)
    for i in range(0, padding_height):
        result = np.concatenate((zero_row, result), axis=0)
        result = np.concatenate((result, zero_row), axis=0)
    for j in range(0, padding_width):
        result = np.concatenate((zero_column, result), axis=1)
        result = np.concatenate((result, zero_column), axis=1)
    return result


# gaussian_kernel = np.array([[1, 2, 1],
#                             [2, 4, 2],
#                             [1, 2, 1]],
#                            dtype=np.float32)
#
# gaussian_kernel = (1 / 16) * gaussian_kernel
# image = cv2.imread('sample_images/im1_1-2.JPG')
# new_image = convolve(image, gaussian_kernel)
# fig = plt.figure(figsize=(10, 10))
# fig.add_subplot(2, 2, 1)
# plt.imshow(image)
# fig.add_subplot(2, 2, 2)
# plt.imshow(new_image)
# plt.show()

# cv2.imshow('image', new_image)
# cv2.imshow('image2', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()