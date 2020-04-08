# CISC642 PR1
Author: Carlton Brady  
Repo link: https://github.com/CarltonTB/Image_Blending  

Python Version: 3.6.3  
Dependencies:  
-opencv  
-matplotlib  
-numpy  

Part 1 results are in the folder called "blending_results"  

Part 2 results are in the folder called "mosaic_results"  

The point correspondences in the affine12.txt files are in the following format:  
[((image1x, image1y),(image2x, image2y)),  
(image1x, image1y),(image2x, image2y))]  

Demos can be run by the following command:  
python3 demo.py  
Demo options 1 through 4 are for part 1, demo option 5 is for part 2  

Part 1 tests can be run by the following command (~50s to run):  
python3 image_blending_tests.py  

Part 2 tests can be run by the following command (~50s to run):  
python3 mosaicing_tests.py  