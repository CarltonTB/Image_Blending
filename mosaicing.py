# Author: Carlton Brady
import image_blending as ib
import numpy as np
import cv2
from matplotlib import pyplot as plt


def mosaic_images_same_size(IA, IB, IAx, IBx, n):
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


def mosiac_with_affine_transformed(IA, transformedB, points, transformation_matrix, n):
    """
    IA is an image of varying size, the image on the left
    transformedB is the image on the right that was transformed with the transformation_matrix
    points is the list of point pairs used to compute the transformation matrix
    transformation matrix is the matrix of affine parameters
    n is the number of pyramid levels to use in blending
    return:
    a new image with the two input images stitched together
    """
    tx = transformation_matrix[0, 2]
    ty = transformation_matrix[1, 2]
    new_height = np.size(transformedB, 0)
    # find the leftmost nonzero pixel by
    zero_column = np.zeros((new_height, 1, 3), dtype=np.uint8)
    min_nonzero_x = int(np.rint(tx))
    for j in range(min_nonzero_x, 0, -1):
        comparison = transformedB[:, j] == zero_column
        if comparison.all():
            min_nonzero_x = j+1
            break
    transformedB_chopped = transformedB[:, min_nonzero_x:]
    # apply padding to image A to make it the same height as the transformedB
    left_image = np.copy(IA)
    if ty > 0:
        left_image = ib.apply_padding(left_image, int(np.rint(ty)), 0, apply_left=False, apply_right=False, apply_top=False)
    else:
        left_image = ib.apply_padding(left_image, int(np.rint(ty)), 0, apply_left=False, apply_right=False, apply_bottom=False)
    right_image = np.copy(transformedB_chopped)
    max_B = 0
    min_A = np.size(IA, 1)
    for point in points:
        if point[0][0] < min_A:
            min_A = point[0][0]
        if point[1][0] > max_B:
            max_B = point[1][0]

    pixel_overlap = (np.size(left_image, 1) - int(np.rint(min_A))) + int(np.rint(max_B))
    new_image_width = np.size(left_image, 1) + np.size(right_image, 1) - pixel_overlap
    assert(np.size(left_image, 0) == np.size(right_image, 0))
    new_image_height = np.size(right_image, 0)
    result = np.zeros((new_image_height, new_image_width, 3))
    result[:, :np.size(left_image, 1)] = left_image
    zero_pixel = np.array([0, 0, 0])
    for i in range(0, new_image_height):
        k = 0
        for j in range(new_image_width-np.size(right_image, 1), new_image_width):
            comparison = right_image[i, k] == zero_pixel
            if not comparison.all():
                result[i, j] = right_image[i, k]
            k += 1
    return result.astype(np.uint8)


def applying_affine_transformation(image, transformation_matrix):
    """
    image is an image of varying size
    transformation_matrix is the matrix of affine parameters
    returns:
    image after affine unwarping has been done
    """
    tx = transformation_matrix[0, 2]
    ty = transformation_matrix[1, 2]
    new_width = np.size(image, 1) + int(abs(np.rint(tx)))
    new_height = np.size(image, 0) + int(abs((np.rint(ty))))
    transformedB = cv2.warpAffine(image, transformation_matrix, dsize=(new_width, new_height))
    return transformedB


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
    points = plt.ginput(num_pairs*2, timeout=5*60)
    pairs = []
    for i in range(0, len(points)-1, 2):
        pair = (points[i], points[i+1])
        pairs.append(pair)
    return pairs


def compute_affine_parameters(I1, I2, point_pairs):
    """
    I1 is an image of varying size
    I2 is an image of varying size
    point_pairs is a list of corresponding points between the images in the form:
    [((I1x, I1y), (I2x, I2y)), ((I1x, I1y), (I2x, I2y))...]
    returns:
    The affine transformation parameters that would warp image I2 into the same view as image I1
    in matrix form like so:
    [[m1, m2, tx],
    [m3, m4, ty]]
    """
    n = len(point_pairs)
    A = np.zeros((2*n, 6))
    j = 0
    for i in range(0, n*2, 2):
        A[i, 0] = point_pairs[j][0][0]
        A[i, 1] = point_pairs[j][0][1]
        A[i, 4] = 1
        A[i+1, 2] = point_pairs[j][0][0]
        A[i+1, 3] = point_pairs[j][0][1]
        A[i+1, 5] = 1
        j += 1
    b = np.zeros((2*n, 1))
    j = 0
    for k in range(0, n*2, 2):
        b[k, 0] = point_pairs[j][1][0]
        b[k+1, 0] = point_pairs[j][1][1]
        j += 1
    # if A is square, we can just do the regular inverse A^-1 * b
    if np.size(A, 0) == np.size(A, 1):
        x = np.matmul(np.linalg.inv(A), b)
    # solve it using the pseudo inverse
    else:
        A_copy = np.copy(A)
        AT = A_copy.transpose()
        ATA = np.matmul(AT, A)
        ATA_inv = np.linalg.inv(ATA)
        psuedo_inv = np.matmul(ATA_inv, A.transpose())
        x = np.matmul(psuedo_inv, b)
    transformation_matrix = np.zeros((2, 3))
    transformation_matrix[0, 0] = x[0, 0]
    transformation_matrix[0, 1] = x[1, 0]
    transformation_matrix[0, 2] = x[4, 0]
    transformation_matrix[1, 0] = x[2, 0]
    transformation_matrix[1, 1] = x[3, 0]
    transformation_matrix[1, 2] = x[5, 0]
    return transformation_matrix


def swap_correspondence_order(pairs):
    """
    pairs is a list of tuples of tuples that represent points correspondences
    returns:
    a new list of tuples of tuples with the points in opposite order
    """
    swapped = []
    for pair in pairs:
        swapped.append((pair[1], (pair[0])))
    return swapped


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

