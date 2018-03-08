The first few functions deal with laplacian blendings

The next few deal with calculating the homography between points in the two images and then warping one of the images and blending with the other. This next part of code also has a function that takes an image, four points, and rectifies the image so the four points make the shape of a rectangle. 

The code outside of the functions in main.py:
The first set gets you the rectification of certain images in the unsorted folder. The code requires your input on what you want rectified into a rectangle. You plot your points using ginput, and remember to input them clockwise starting from the top left hand corner. 

In between is code commented out in case you want to give your own points of the images we will morph together 

The second set includes points pre-plotted on the given pictures, and the code to take these points and images and warp and blend the images together. This code makes a mosaic
