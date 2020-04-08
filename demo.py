# Author: Carlton Brady
import cv2
import numpy as np
import image_blending as ib
import mosaicing as mos


class ImageBlendingDemo:

    def __init__(self):
        self.image_a_coordinates = None
        self.image_b_coordinates = None

    def click_a(self, event, x, y, flags, params):
        """
        x is the x coordinate in the image where the user clicked
        y is the y coordinate in the image where the user clicked
        returns:
        a tuple containing coordinates (x, y)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_a_coordinates = (x, y)
            print("ImageA selected coordinates =", self.image_a_coordinates)
            return x, y

    def click_b(self, event, x, y, flags, params):
        """
        x is the x coordinate in the image where the user clicked
        y is the y coordinate in the image where the user clicked
        returns:
        a tuple containing coordinates (x, y)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_b_coordinates = (x, y)
            print("ImageB selected coordinates =", self.image_b_coordinates)
            return x, y

    def run_blending_demo(self):
        print("Click any point in imageA. A bit mask will be created with\n"
              " 1's to the right of the x coordinate where you clicked")
        print("Click on either image and press any key to close the images and continue with blending")
        # The two images being used for the demo, where imageA is the left image and imageB is the right image
        imageA = cv2.imread('sample_images/apple.jpg')
        imageB = cv2.imread('sample_images/orange.jpg')
        # imageA = cv2.imread('sample_images/im1_1-2.jpg')
        # imageB = cv2.imread('sample_images/im1_2-1.jpg')
        # imageA = cv2.imread('sample_images/im2_3.JPG')
        # imageB = cv2.imread('sample_images/im2_2-1.JPG')
        # imageA = cv2.imread('sample_images/im5_1.png')
        # imageB = cv2.imread('sample_images/im5_2.png')
        cv2.namedWindow("imageA")
        cv2.setMouseCallback("imageA", demoObj.click_a)
        cv2.imshow('imageA', imageA)
        cv2.namedWindow("imageB")
        cv2.setMouseCallback("imageB", demoObj.click_b)
        cv2.imshow('imageB', imageB)
        cv2.waitKey(0)
        levels = input("Enter the number of pyramid levels you want to use for blending (3-10 ideally)\n")
        levels = int(levels)
        x_coord = self.image_a_coordinates[0]
        print("pixel overlap =", np.size(imageA, 1) - x_coord)
        print("Creating bitmask...")
        bitmask = ib.create_bitmask(np.size(imageA, 0), np.size(imageA, 1), np.size(imageA, 2),
                                    (0, x_coord), np.size(imageA, 0), np.size(imageA, 1)-x_coord)
        cv2.imshow('bitmask', np.rint(255*bitmask).astype(np.uint8))
        print("Bitmask complete!")
        print("Click on any image and press any key to close the images and continue with blending")
        cv2.waitKey(0)
        print("Blending images...")
        blended_image = ib.blend_images(imageB, imageA, bitmask, levels)
        print("Blending complete!")
        print("Click on either image and press any key to close the images and end the demo")
        cv2.imshow("blended", blended_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Demo Complete")

    def run_affine_and_mosaic_demo(self):
        imageA = cv2.imread("sample_images/im4_1.png")
        imageB = cv2.imread("sample_images/im4_2.png")
        print("Select 3 pairs of corresponding points in the images in alternating order by first clicking \n" 
              "on the left image, then clicking on its corresponding point in the right image. Repeat twice more.")
        # pairs = mos.select_control_points(imageA, imageB)
        pairs = [((548.1038961038961, 511.1428571428571), (9.681818181818244, 512.0714285714284)),
                 ((573.1753246753246, 451.71428571428555), (30.110389610389802, 450.7857142857142)),
                 ((562.961038961039, 370.92857142857133), (20.82467532467558, 370.0))]
        print(pairs)
        inverse_transformation_matrix = mos.compute_affine_parameters(imageA, imageB, pairs)
        transformation_matrix = mos.compute_affine_parameters(imageA, imageB, mos.swap_correspondence_order(pairs))
        print("affine params:\n", inverse_transformation_matrix)
        print("inverse affine params:\n", transformation_matrix)
        transformedB = mos.applying_affine_transformation(imageB, transformation_matrix)
        cv2.imshow('imageB', imageB)
        cv2.imshow('transformedB', transformedB)
        cv2.imshow('imageA', imageA)
        result = mos.mosiac_with_affine_transformed(imageA, transformedB, pairs, transformation_matrix, 2)
        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Demo Complete")

    def run_gaussian_pyramid_demo(self):
        demo_image = cv2.imread('sample_images/im1_1-2.JPG')
        levels = input("Enter the number of Gaussian pyramid levels you want (3-10 ideally)\n")
        levels = int(levels)
        print("Creating Guassian pyramid...")
        g_pyramid = ib.gaussian_pyramid(demo_image, levels)
        for i in range(0, len(g_pyramid)):
            cv2.imshow('Level ' + str(i), np.rint(g_pyramid[i]).astype(np.uint8))
        print("Gaussian pyramid complete!")
        print("Click on either image and press any key to close the images and end the demo")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Demo Complete")

    def run_laplacian_pyramid_demo(self):
        demo_image = cv2.imread('sample_images/apple.jpg')
        levels = input("Enter the number of Laplacian pyramid levels you want (3-10 ideally)\n")
        levels = int(levels)
        print("Creating Laplacian pyramid...")
        l_pyramid = ib.laplacian_pyramid(demo_image, levels)
        for i in range(0, len(l_pyramid)):
            cv2.imshow('Level ' + str(i), np.rint(l_pyramid[i]).astype(np.uint8))
        print("Laplacian pyramid complete!")
        print("Click on either image and press any key to close the images and end the demo")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Demo Complete")

    def run_reconstruction_demo(self):
        original_image = cv2.imread('sample_images/im1_1-2.JPG')
        levels = input("Enter the number of Laplacian pyramid levels you want (3-10 ideally)\n")
        levels = int(levels)
        print("Creating Laplacian pyramid...")
        l_pyramid = ib.laplacian_pyramid(original_image, levels)
        print("Laplacian pyramid complete!")
        reconstructed = ib.reconstruct(l_pyramid)
        cv2.imshow("original", original_image)
        cv2.imshow("reconstructed", np.rint(reconstructed).astype(np.uint8))
        # Reconstruction error
        oi = original_image.astype(np.float32)
        rec = reconstructed.astype(np.float32)
        print("reconstruction error: ", oi-rec)
        total_reconstruction_error = np.sum(((oi-rec)**2)**(1/2))
        print("Total reconstruction error: " + str(total_reconstruction_error))
        print("Click on either image and press any key to close the images and end the demo")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Demo Complete")


if __name__ == "__main__":
    demoObj = ImageBlendingDemo()
    choice = input("Demo options (enter the corresponding number for the desired demo):\n"
                   "1) Image blending\n"
                   "2) Gaussian pyramid\n"
                   "3) Laplacian pyramid\n"
                   "4) Image reconstruction\n"
                   "5) Affine unwarping and mosaicing\n")
    if choice == "1":
        demoObj.run_blending_demo()
    elif choice == "2":
        demoObj.run_gaussian_pyramid_demo()
    elif choice == "3":
        demoObj.run_laplacian_pyramid_demo()
    elif choice == "4":
        demoObj.run_reconstruction_demo()
    elif choice == "5":
        demoObj.run_affine_and_mosaic_demo()
