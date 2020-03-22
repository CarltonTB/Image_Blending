# Author: Carlton Brady
import cv2
from matplotlib import pyplot as plt
import numpy as np


def convolve(I, H):
    """
    I is an image of varying size
    H is a kernel of varying size
    both should be numpy arrays
    returns:
    A new image that is the result of the convolution
    """
    # Stride length is assumed to be 1
    # Assume no padding for now
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
    output_width = image_width - filter_width + 1
    output_height = image_height - filter_height + 1
    iterations = 0
    result_list = []
    for i in range(0, output_height):
        new_row_list = []
        for j in range(0, output_width):
            new_channel_list = []
            for k in range(0, image_channels):
                image_slice = I[i:i+filter_height, j:j+filter_width, k]
                dot_product = np.sum(H*image_slice)
                dot_product = int(round(dot_product, 0))
                new_channel_list.append(dot_product)
                # print(image_slice)
                iterations += 1
            new_row_list.append(new_channel_list)
        result_list.append(new_row_list)
    result = np.array(result_list)
    return result.astype(np.uint8)


kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]],
                  dtype=np.float32)
zeros = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]],
                 dtype=np.float32)
test = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                 [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                 [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                 ])
print(test[0:3, 0:3, 0])
kernel = (1/16)*kernel
print(kernel)
image = cv2.imread('sample_images/im1_1-2.JPG')
new_image = convolve(image, kernel)
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(2, 2, 1)
plt.imshow(image)
fig.add_subplot(2, 2, 2)
plt.imshow(new_image)
plt.show()
# cv2.imshow('image', new_image)
# cv2.imshow('image2', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()