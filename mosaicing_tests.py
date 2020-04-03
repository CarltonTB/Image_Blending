# Author: Carlton Brady
import unittest
import mosaicing as mos
import numpy as np
import cv2
from matplotlib import pyplot as plt


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

    def test_mosaicing(self):
        imageA = cv2.imread('sample_images/im4_1.png')
        imageB = cv2.imread('sample_images/im4_2.png')
        mosaic = mos.mosaic_images(imageA, imageB, 599, 66, 3)
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


if __name__ == '__main__':
    unittest.main()
