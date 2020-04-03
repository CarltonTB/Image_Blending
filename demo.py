# Author: Carlton Brady
import cv2
import numpy as np
import image_blending as ib


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
        print("Click any point in the images. A bit mask will be created with\n"
              " 1's to the right of the x coordinate where you clicked")
        print("Click on either image and press any key to close the images and continue with blending")
        imageA = cv2.imread('sample_images/apple.jpg')
        imageB = cv2.imread('sample_images/orange.jpg')
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
        # TODO: figure out why this is zero
        oi = original_image.astype(np.float32)
        rec = reconstructed.astype(np.float32)
        print(oi-rec)
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
                   "4) Image reconstruction\n")
    if choice == "1":
        demoObj.run_blending_demo()
    elif choice == "2":
        demoObj.run_gaussian_pyramid_demo()
    elif choice == "3":
        demoObj.run_laplacian_pyramid_demo()
    elif choice == "4":
        demoObj.run_reconstruction_demo()
