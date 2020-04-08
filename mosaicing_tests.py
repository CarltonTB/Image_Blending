# Author: Carlton Brady
import unittest
import mosaicing as mos
import numpy as np
import cv2
from matplotlib import pyplot as plt


# Tests take about 40-50 seconds to run
class MosaicingTests(unittest.TestCase):

    def test_create_mosaicing_bitmask(self):
        bitmask = mos.create_mosaic_bitmask(400, 600, 3, (0, 400), 200, (0, 200), 200)
        bitmask = np.rint(255*bitmask).astype(np.uint8)
        self.assertEqual(np.size(bitmask, 0), 400)
        self.assertEqual(np.size(bitmask, 1), 600)
        self.assertEqual(np.size(bitmask, 2), 3)
        # For visual debugging
        # cv2.imshow('bitmask', bitmask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_swap_correspondence_order(self):
        pairs = [((1, 2), (3, 4)), ((1, 2), (3, 4)), ((1, 2), (3, 4))]
        swapped = mos.swap_correspondence_order(pairs)
        self.assertEqual(swapped, [((3, 4), (1, 2)), ((3, 4), (1, 2)), ((3, 4), (1, 2))])

    def test_mosaicing(self):
        imageA = cv2.imread('sample_images/im4_1.png')
        imageB = cv2.imread('sample_images/im4_2.png')
        mosaic = mos.mosaic_images_same_size(imageA, imageB, 599, 66, 3)
        self.assertEqual(np.size(mosaic, 0), np.size(imageA, 0))
        self.assertEqual(np.size(mosaic, 0), np.size(imageB, 0))
        # For visual debugging
        # cv2.imshow('mosaic', mosaic)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def test_select_points(self):
        imageA = cv2.imread("sample_images/mountain_A.jpg")
        imageB = cv2.imread("sample_images/mountain_B.jpg")
        self.assertEqual(np.size(imageA, 2), 3)
        self.assertEqual(np.size(imageB, 2), 3)
        # For visual debugging
        # pairs = mos.select_control_points(imageA, imageB, 2)
        # self.assertEqual(len(pairs), 2)

    def test_compute_affine_parameters(self):
        imageA = cv2.imread("sample_images/mountain_A.jpg")
        imageB = cv2.imread("sample_images/mountain_B.jpg")
        # uncomment the below line if you want to manually choose different points
        # pairs = mos.select_control_points(imageA, imageB, 3)
        # comment out the below line if you want to manually choose different points
        pairs = [((378.87793548387094, 143.93922580645176), (112.17264516129046, 127.79548387096793)), ((351.72709677419357, 138.80258064516147), (82.82038709677431, 116.78838709677439)), ((466.9347096774194, 157.14774193548396), (192.15754838709688, 157.14774193548396))]
        # print(pairs)
        ptsA = []
        ptsB = []
        for i in range(0, len(pairs)):
            ptsA.append([pairs[i][0][0], pairs[i][0][1]])
            ptsB.append([pairs[i][1][0], pairs[i][1][1]])
        ptsA = np.float32(ptsA)
        ptsB = np.float32(ptsB)
        # check to see if my matrix math gives the same answer as built in opencv function
        M = cv2.getAffineTransform(ptsA, ptsB)
        transformation_matrix = mos.compute_affine_parameters(imageA, imageB, pairs)
        # print(transformation_matrix)
        # print(M)
        self.assertTrue(np.allclose(M, transformation_matrix, atol=0.0001))

    def test_compute_affine_parameters_overconstrained(self):
        imageA = cv2.imread("sample_images/mountain_A.jpg")
        imageB = cv2.imread("sample_images/mountain_B.jpg")
        # uncomment the below line if you want to manually choose different points
        # pairs = mos.select_control_points(imageA, imageB, 4)
        # comment out the below line if you want to manually choose different points
        pairs = [((378.1441290322581, 142.47161290322583), (111.4388387096775, 124.8602580645163)), ((358.3313548387097, 144.67303225806472), (93.093677419355, 123.39264516129037)), ((469.13612903225805, 158.6153548387099), (193.6251612903227, 160.81677419354855)), ((433.1796129032258, 187.96761290322593), (154.73341935483882, 178.42812903225808))]
        # print(pairs)
        ptsA = []
        ptsB = []
        for i in range(0, 3):
            ptsA.append([pairs[i][0][0], pairs[i][0][1]])
            ptsB.append([pairs[i][1][0], pairs[i][1][1]])
        ptsA = np.float32(ptsA)
        ptsB = np.float32(ptsB)
        # check to see if my matrix math gives the same answer as built in opencv function
        M = cv2.getAffineTransform(ptsA, ptsB)
        transformation_matrix = mos.compute_affine_parameters(imageA, imageB, pairs)
        # print(transformation_matrix)
        # print(M)
        self.assertTrue(np.allclose(M, transformation_matrix, atol=30))

    def test_apply_affine_transformation(self):
        imageA = cv2.imread("sample_images/im4_1.png")
        imageB = cv2.imread("sample_images/im4_2.png")
        # uncomment the below line if you want to manually choose different points
        # pairs = mos.select_control_points(imageA, imageB, 3)
        # comment out the below line if you want to manually choose different points
        pairs = [((548.1038961038961, 511.1428571428571), (9.681818181818244, 512.0714285714284)), ((573.1753246753246, 451.71428571428555), (30.110389610389802, 450.7857142857142)), ((562.961038961039, 370.92857142857133), (20.82467532467558, 370.0))]
        print(pairs)
        inverse_transformation_matrix = mos.compute_affine_parameters(imageA, imageB, pairs)
        transformation_matrix = mos.compute_affine_parameters(imageA, imageB, mos.swap_correspondence_order(pairs))
        print("affine params:\n", transformation_matrix)
        print("inverse affine params:\n", inverse_transformation_matrix)
        transformedB = mos.applying_affine_transformation(imageB, transformation_matrix)
        self.assertEqual(np.size(transformedB, 0), np.size(imageB, 0)+2)
        result = mos.mosiac_with_affine_transformed(imageA, transformedB, pairs, transformation_matrix, 2)
        # for visual debugging
        # cv2.imshow('imageB', imageB)
        # cv2.imshow('transformedB', transformedB)
        # cv2.imshow('result', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
