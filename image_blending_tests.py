# Author: Carlton Brady
import unittest
import image_blending as ib
import numpy as np
import cv2
from matplotlib import pyplot as plt


# tests take about 40-50s seconds to run
class ImageBlendingTests(unittest.TestCase):

    def test_apply_padding(self):
        test = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                         [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                         [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                         ])
        test_output1 = ib.apply_padding(test, 1, 1)
        # check height
        self.assertEqual(np.size(test_output1, 0), 5)
        # check width
        self.assertEqual(np.size(test_output1, 1), 5)
        # check channels/depth
        self.assertEqual(np.size(test_output1, 2), 3)
        test_image = cv2.imread('sample_images/im1_1-2.JPG')
        test_output2 = ib.apply_padding(test_image, 1, 1)
        self.assertEqual(np.size(test_output2, 0), 302)
        self.assertEqual(np.size(test_output2, 1), 402)
        self.assertEqual(np.size(test_output2, 2), 3)
        test_output3 = ib.apply_padding(test_image, 2, 2)
        self.assertEqual(np.size(test_output3, 0), 304)
        self.assertEqual(np.size(test_output3, 1), 404)
        self.assertEqual(np.size(test_output3, 2), 3)

    def test_convolution(self):
        test_image = cv2.imread('sample_images/im1_1-2.JPG')
        gaussian_kernel = np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]],
                                   dtype=np.float32)
        gaussian_kernel = (1 / 16) * gaussian_kernel
        blurred_image = ib.convolve(test_image, gaussian_kernel)
        self.assertEqual(np.size(blurred_image, 0), np.size(test_image, 0))
        self.assertEqual(np.size(blurred_image, 1), np.size(test_image, 1))
        self.assertEqual(np.size(blurred_image, 2), np.size(test_image, 2))
        # For visual debugging
        # convert from BGR to RGB for display
        # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        # blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
        # fig = plt.figure(figsize=(10, 10))
        # fig.add_subplot(2, 2, 1)
        # plt.imshow(test_image.astype(np.uint8))
        # fig.add_subplot(2, 2, 2)
        # plt.imshow(blurred_image.astype(np.uint8))
        # plt.show()

    def test_1D_separable_gaussian_blurring(self):
        test_image = cv2.imread('sample_images/im1_1-2.JPG')
        row_gaussian_kernel = np.array([[1, 2, 1]],
                                       dtype=np.float32)
        row_gaussian_kernel = (1 / 4) * row_gaussian_kernel
        col_gaussian_kernel = row_gaussian_kernel.transpose()
        col_kernel_blurred_image = ib.convolve(test_image, col_gaussian_kernel)
        self.assertEqual(np.size(col_kernel_blurred_image, 0), np.size(test_image, 0))
        self.assertEqual(np.size(col_kernel_blurred_image, 1), np.size(test_image, 1))
        self.assertEqual(np.size(col_kernel_blurred_image, 2), np.size(test_image, 2))
        blurred_image = ib.convolve(col_kernel_blurred_image, row_gaussian_kernel)
        self.assertEqual(np.size(blurred_image, 0), np.size(test_image, 0))
        self.assertEqual(np.size(blurred_image, 1), np.size(test_image, 1))
        self.assertEqual(np.size(blurred_image, 2), np.size(test_image, 2))
        # For visual debugging
        # convert from BGR to RGB for display
        # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        # blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
        # fig = plt.figure(figsize=(10, 10))
        # fig.add_subplot(2, 2, 1)
        # plt.imshow(test_image.astype(np.uint8))
        # fig.add_subplot(2, 2, 2)
        # plt.imshow(blurred_image.astype(np.uint8))
        # plt.show()

    def test_reduce(self):
        test_image = cv2.imread('sample_images/im1_1-2.JPG')
        smaller_image = ib.reduce(test_image)
        self.assertEqual(np.size(smaller_image, 0), np.size(test_image, 0)/2)
        self.assertEqual(np.size(smaller_image, 1), np.size(test_image, 1)/2)
        self.assertEqual(np.size(smaller_image, 2), np.size(test_image, 2))
        # For visual debugging
        # cv2.imshow('original', test_image.astype(np.uint8))
        # cv2.imshow('reduced', smaller_image.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_expand(self):
        test_image = cv2.imread('sample_images/im1_1-2.JPG')
        larger_image = ib.expand(test_image)
        self.assertEqual(np.size(larger_image, 0), np.size(test_image, 0)*2)
        self.assertEqual(np.size(larger_image, 1), np.size(test_image, 1)*2)
        self.assertEqual(np.size(larger_image, 2), np.size(test_image, 2))
        # For visual debugging
        # cv2.imshow('original', test_image.astype(np.uint8))
        # cv2.imshow('expanded', larger_image.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_gaussian_pyramid(self):
        test_image = cv2.imread('sample_images/im1_1-2.JPG')
        g_pyramid = ib.gaussian_pyramid(test_image, 3)
        self.assertEqual(len(g_pyramid), 3)
        self.assertEqual(np.size(g_pyramid[0], 0), np.size(test_image, 0))
        self.assertEqual(np.size(g_pyramid[0], 1), np.size(test_image, 1))
        self.assertEqual(np.size(g_pyramid[0], 2), np.size(test_image, 2))
        self.assertEqual(np.size(g_pyramid[1], 0), np.size(test_image, 0)/2)
        self.assertEqual(np.size(g_pyramid[1], 1), np.size(test_image, 1)/2)
        self.assertEqual(np.size(g_pyramid[1], 2), np.size(test_image, 2))
        self.assertEqual(np.size(g_pyramid[2], 0), np.size(test_image, 0)/4)
        self.assertEqual(np.size(g_pyramid[2], 1), np.size(test_image, 1)/4)
        self.assertEqual(np.size(g_pyramid[2], 2), np.size(test_image, 2))
        # For visual debugging
        # cv2.imshow('image1', g_pyramid[0].astype(np.uint8))
        # cv2.imshow('image2', g_pyramid[1].astype(np.uint8))
        # cv2.imshow('image3', g_pyramid[2].astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_laplacian_pyramid(self):
        test_image = cv2.imread('sample_images/apple.jpg')
        l_pyramid = ib.laplacian_pyramid(test_image, 4)
        self.assertEqual(np.size(l_pyramid[0], 0), np.size(test_image, 0))
        self.assertEqual(np.size(l_pyramid[0], 1), np.size(test_image, 1))
        self.assertEqual(np.size(l_pyramid[0], 2), np.size(test_image, 2))
        self.assertEqual(round(np.size(l_pyramid[1], 0), 0), round(np.size(test_image, 0)/2, 0))
        self.assertEqual(round(np.size(l_pyramid[1], 1), 0), round(np.size(test_image, 1)/2, 0))
        self.assertEqual(np.size(l_pyramid[1], 2), np.size(test_image, 2))
        self.assertEqual(round(np.size(l_pyramid[2], 0), 0), round(np.size(test_image, 0)/4, 0))
        self.assertEqual(round(np.size(l_pyramid[2], 1), 0), round(np.size(test_image, 1)/4, 0))
        self.assertEqual(np.size(l_pyramid[2], 2), np.size(test_image, 2))
        self.assertEqual(round(np.size(l_pyramid[3], 0), 0), round(np.size(test_image, 0)/8, 0))
        self.assertEqual(round(np.size(l_pyramid[3], 1), 0), round(np.size(test_image, 1)/8, 0))
        self.assertEqual(np.size(l_pyramid[3], 2), np.size(test_image, 2))
        # For visual debugging
        # cv2.imshow('image1', l_pyramid[0].astype(np.uint8))
        # cv2.imshow('image2', l_pyramid[1].astype(np.uint8))
        # cv2.imshow('image3', l_pyramid[2].astype(np.uint8))
        # cv2.imshow('image4', l_pyramid[3].astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_reconstruction(self):
        test_image = cv2.imread('sample_images/im1_1-2.JPG')
        l_pyramid = ib.laplacian_pyramid(test_image, 4)
        reconstructed = ib.reconstruct(l_pyramid)
        self.assertEqual(np.size(reconstructed, 0), np.size(test_image, 0))
        self.assertEqual(np.size(reconstructed, 1), np.size(test_image, 1))
        self.assertEqual(np.size(reconstructed, 2), np.size(test_image, 2))
        # For visual debugging
        # cv2.imshow('original', test_image.astype(np.uint8))
        # cv2.imshow('reconstructed', reconstructed.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_create_bitmask(self):
        test_image = cv2.imread('sample_images/im1_1-2.JPG')
        bitmask = ib.create_bitmask(np.size(test_image, 0), np.size(test_image, 1), np.size(test_image, 2),
                                    (0, 0), np.size(test_image, 0), int(np.size(test_image, 1)/2))
        self.assertEqual(np.size(bitmask, 0), np.size(test_image, 0))
        self.assertEqual(np.size(bitmask, 1), np.size(test_image, 1))
        self.assertEqual(np.size(bitmask, 2), np.size(test_image, 2))
        # For visual debugging
        # bitmask = 255*bitmask
        # cv2.imshow("mask", bitmask.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_blend_images(self):
        test_image_A = cv2.imread('sample_images/apple.JPG')
        test_image_B = cv2.imread('sample_images/orange.JPG')
        blended_pyramid = ib.blend_images(test_image_B, test_image_A, 4)
        type_example = type(np.ones((1, 1, 3)))
        self.assertEqual(type(blended_pyramid[0]), type_example)
        self.assertEqual(type(blended_pyramid[1]), type_example)
        self.assertEqual(type(blended_pyramid[2]), type_example)
        self.assertEqual(type(blended_pyramid[3]), type_example)
        blended_image = ib.reconstruct(blended_pyramid)
        self.assertEqual(np.size(blended_image, 0), np.size(test_image_A, 0))
        self.assertEqual(np.size(blended_image, 1), np.size(test_image_A, 1))
        self.assertEqual(np.size(blended_image, 2), np.size(test_image_A, 2))
        self.assertEqual(np.size(blended_image, 0), np.size(test_image_B, 0))
        self.assertEqual(np.size(blended_image, 1), np.size(test_image_B, 1))
        self.assertEqual(np.size(blended_image, 2), np.size(test_image_B, 2))
        # For visual debugging
        # cv2.imshow('image A', test_image_A.astype(np.uint8))
        # cv2.imshow('image B', test_image_B.astype(np.uint8))
        # cv2.imshow('blended', blended_image.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
