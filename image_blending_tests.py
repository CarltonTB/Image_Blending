import unittest
import image_blending as ib
import numpy as np
import cv2


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


if __name__ == '__main__':
    unittest.main()
