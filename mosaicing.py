# Author: Carlton Brady
import image_blending as ib
import numpy as np
import cv2
from matplotlib import pyplot as plt


def mosaic_images(IA, IB, IAx, IBx, n):
    """
    This function mosaics images with an overlapping region spanning between
    the x coordinate chosen on Image A and the x coordinate chosen on Image B
    IA is an image of varying size
    IB is an image of varying size
    IAx is the x coordinate selected on image A
    IBx is the x coordinate selected on image B
    n is the number of Laplacian pyramid levels to use in blending
    returns:
    a new image that is the 2 input images blended together with the given overlapping region
    """
    assert(np.size(IA, 0) == np.size(IB, 0))
    new_image_width = (np.size(IA, 1) + np.size(IB, 1)) - (IBx + (np.size(IA, 1)-IAx))
    left_image = np.copy(IA)
    right_image = np.copy(IB)
    left_image = ib.apply_padding(left_image, 0, new_image_width - np.size(IA, 1), apply_left=False)
    right_image = ib.apply_padding(right_image, 0, new_image_width - np.size(IB, 1), apply_right=False)
    blend_region_width = (IBx + (np.size(IA, 1)-IAx))
    ones_region_width = new_image_width - (IAx + blend_region_width)
    # bitmask = create_mosaic_bitmask(np.size(IA, 0), new_image_width, np.size(IA, 2),
    #                                 (0, IAx + blend_region_width), ones_region_width, (0, IAx), blend_region_width)
    bitmask = ib.create_bitmask(np.size(IA, 0), new_image_width, np.size(IA, 2),
                                (0, IAx), np.size(IA, 0), new_image_width-IAx)
    assert(np.size(left_image) == np.size(right_image) == np.size(bitmask))
    # For visual debugging
    # cv2.imshow('left', left_image.astype(np.uint8))
    # cv2.imshow('right', right_image.astype(np.uint8))
    # cv2.imshow('bitmask', np.rint(255*bitmask).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    blended = ib.blend_images(right_image, left_image, bitmask, n)
    return blended


def select_control_points(imageA, imageB, num_pairs):
    """
    imageA is an image of varying size
    imageB is an image of varying size
    returns:
    a list of tuples of tuples containing the pairs of points selected by the user.
    e.g. [((x1, y1),(x2, y2)), ((x1, y1),(x2, y2))]
    """
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(2, 2, 1)
    plt.imshow(imageA)
    fig.add_subplot(2, 2, 2)
    plt.imshow(imageB)
    points = plt.ginput(num_pairs*2)
    pairs = []
    for i in range(0, len(points)-1, 2):
        pair = (points[i], points[i+1])
        pairs.append(pair)
    return pairs


def create_mosaic_bitmask(height, width, depth, ones_anchor_point, ones_region_width,
                          blend_anchor_point, blend_region_width):
    """
    height is the height of the bitmask
    width is the width if the bitmask to be created
    depth is the number of channels in the bitmask to be created
    ones_anchor_point is the coordinates of where the onces region will start
    ones_region_width is the width of the ones region
    blend_anchor_point is the coordinates of where the blend (0.5) region will start
    blend_region_width is the width of the blend region
    returns:
    a bitmask matching the specific parameters
    """
    bitmask = np.zeros((height, width, depth), dtype=np.float32)
    ones_region = np.ones((height, ones_region_width, depth), dtype=np.float32)
    blend_region = np.ones((height, blend_region_width, depth), dtype=np.float32)/2
    bitmask[ones_anchor_point[0]:ones_anchor_point[0]+height, ones_anchor_point[1]:ones_anchor_point[1]+ones_region_width] = ones_region
    bitmask[blend_anchor_point[0]:blend_anchor_point[0]+height, blend_anchor_point[1]:blend_anchor_point[1]+blend_region_width] = blend_region
    return bitmask

