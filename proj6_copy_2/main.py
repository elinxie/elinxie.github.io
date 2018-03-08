
# coding: utf-8

# In[1]:


# tools to go blend objects

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
import math
from skimage import filters

# here let's do the homography function and get it out of the way
from numpy.linalg import lstsq, inv
get_ipython().magic(u'pylab')
import math
import matplotlib.pyplot as plt
import skimage.transform as sktr
import scipy as sp
from skimage.draw import polygon
from scipy.interpolate import RectBivariateSpline, interp2d
from numpy.linalg import inv
from scipy.spatial import Delaunay
import os
from itertools import combinations

    
def gaussian_stack(im, N, sigma=1):
    my_im = im
    stack_lst = []
    for i in range(0,N):
        my_im = np.clip(filters.gaussian(my_im, sigma),0,1)
        stack_lst += [my_im]
    return stack_lst

def laplace_stack(im, N, sigma=1):
    a_im = im
    stack_lst = []
    for j in range(0,N - 1):
        a_im = a_im - filters.gaussian(a_im, sigma)
        stack_lst += [a_im]
    stack_lst += [gaussian_stack(im,5)[-1]]
    return stack_lst
def make2dhybrid(im1, im2, im_m):
    im1_stack = laplace_stack(im1,2)
    im2_stack = laplace_stack(im2,2)
    im_mask_stack = gaussian_stack(im_m,2,10)
    return makeLS_stack(im1_stack, im2_stack, im_mask_stack)
    
def makeLS_stack(LA_s, LB_s, GR_s):
    LS = np.zeros((len(LA_s[0]),len(LA_s[0][0]),1))
    for LA, LB, GR in zip(LA_s,LB_s, GR_s):
        LS += GR*LA
        LS += (1-GR)*LB
    return np.clip(LS,0,1)
    
def makeblack(img):
    LS = img
    for i in range(0, len(LA)):
        for j in range(0,len(LA[0])):
            g_int = LA[i][j]
            print(g_int)  
    return LS

def multi_blend(im1, im2, im_mask):
    [im1_r, im1_g, im1_b] = np.dsplit(im1,3)
    [im2_r, im2_g, im2_b] = np.dsplit(im2,3)    
    [im_m, im_m2, im_m3] = np.dsplit(im_mask, 3)
    hybrid_r = make2dhybrid(im1_r, im2_r, im_m)
    hybrid_g = make2dhybrid(im1_g, im2_g, im_m)
    hybrid_b = make2dhybrid(im1_b, im2_b, im_m)
    hybrid = np.dstack((hybrid_r, hybrid_g, hybrid_b))
    print(hybrid_r.shape)
    skio.imshow(hybrid)
    skio.show()
    return hybrid

    


# In[2]:


# what do I do? Do I just grab the points in a plot?

# magical point-grabbing function here that I used before


# In[3]:




def dist(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_avg_dist(points):
    distances = [dist(p1, p2) for p1, p2 in combinations(points, 2)]
    return sum(distances) / len(distances)

def get_pts(im, num_points=36):
    plt.imshow(im)
    x = plt.ginput(num_points, timeout=200)
    plt.close()
    return x

def get_points(im1, im2,num_points=36):
    print('Please select 2 points in each image for alignment.')
    x = get_pts(im1, num_points)
    y = get_pts(im2, num_points)
    return (x,y)

def computeH(im1_pts, im2_pts):
    if (len(im1_pts) != len(im2_pts)):
        return "hehe get your points aligned"
#     build b
    b = []
    A = []
#     build my A
    for p1, p2 in zip(im1_pts, im2_pts):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        n_row_1 = [x1, y1, 1, 0 ,0 ,0, -x1*x2, -y1*x2]
        n_row_2 = [0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2]
        A.append(n_row_1)
        A.append(n_row_2)
        b.append(x2)
        b.append(y2)
    A = np.array(A)
    b = np.array(b)
#     find h

    A_T = np.transpose(A)
#     h = np.dot(np.dot(inv(np.dot(A_T, A)),A_T),b)
    h = lstsq(A,b)[0]

    h = np.append(h, 1)

    H = np.reshape(h, (3,3))
    return H

# so we have negative number, goal is to transform the matrix of the homography and then we can give shape after transformation
# this is so we get the full image.
# just shift by y
def get_H_shape(im_height, im_width, H):
#     just get the h shape first then do the rectification
    corners = [[0,0,1],
                [im_width, 0,1],
                [im_width,im_height,1],
                [0, im_height, 1]]

    corners_T = [np.array(c).T for c in corners]
    new_corners = [H.dot(cn) for cn in corners_T]
    print(new_corners)
    to_examine_x = [n[0]/n[2] for n in new_corners]
    to_examine_y = [n[1]/n[2] for n in new_corners]
    min_x = min(to_examine_x)
    max_x = max(to_examine_x)
    min_y = min(to_examine_y)
    max_y = max(to_examine_y)
    t_mat = np.identity(3)
    t_mat[1][2] = -min_y
    new_x = int(max(int(math.ceil(max_x - min_x)), max_x))
    new_y = int(max(int(math.ceil(max_y - min_y)), max_y))
    my_shape = (new_y, new_x)
    
    return my_shape, t_mat, min_x

   
# ideally, if we do not want the second image, then we can just have no im2name,
# and then after we just show the rectified image 
# just have a different function
# all this does is just turn a set of four points into a square, and places that square in the center of an image.
# select points COUNTERCLOCKWISE
def rect_image(im1name,  im1_pts=[]):
    print("select points CLOCKWISE from top righthand corner")
    im1 = skio.imread(im1name)/255
    if (im1_pts==[]):
        im1_pts= get_pts(im1,4)
    mid_x = int(len(im1[0])/2)
    mid_y = int(len(im1)/2)
    avg_dist_x = dist(im1_pts[0], im1_pts[1]) + dist(im1_pts[2], im1_pts[3]) / 2
    avg_dist_y = dist(im1_pts[1], im1_pts[2]) + dist(im1_pts[3], im1_pts[0]) / 2
    im2_pts = [[mid_x, mid_y], [mid_x+avg_dist_x, mid_y], [mid_x+avg_dist_x, mid_y+avg_dist_y], [mid_x, mid_y+avg_dist_y]]
    H = computeH(im1_pts, im2_pts)
    my_shape, t_mat, _ = get_H_shape(len(im1), len(im1[0]), H)
    im1_rect = sk.transform.warp(im1, inv(t_mat.dot(H)), output_shape=my_shape)
    plt.imshow(im1_rect)
    skio.imsave('{0}_rect.jpg'.format(im1name[:-4]), np.clip(im1_rect,0,1))
    return im1_rect

# we want a function that can just transform an image based on a few points. 
def make_mosaic(im1name, im2name, im1_pts=[], im2_pts=[],lbl1=None, lbl2=None):
    im1 = skio.imread(im1name)/255
    im2 = skio.imread(im2name)/255
    if (im1_pts==[] or im2_pts==[]):
        im1_pts, im2_pts = get_points(im1, im2,10)
    if (lbl1==None or lbl2==None):
        lbl1 = im1name[-7:-4]
        lbl2 = im2name[-7:-4]
    H = computeH(im1_pts, im2_pts)
    print(len(im1), print(len(im1[0])))
    
    my_shape, t_mat, min_x = get_H_shape(len(im1), len(im1[0]), H)
#     my_shape = (10000,10000)
    print(my_shape)
    im1_rect = sk.transform.warp(im1, inv(t_mat.dot(H)), output_shape=my_shape)
    
#     make naive
    im2_rect = sk.transform.warp(im2, inv(t_mat), output_shape=(my_shape))
    im1_mask = sk.transform.warp(im1, inv(H))
    im1_mask =  sk.transform.warp(im1_mask, inv(t_mat), output_shape=my_shape)
    im2_new = im2_rect - im1_mask
    im_new = im2_new + im1_rect
    plt.imshow(im_new)
    skio.imsave('{0}_{1}_naive.jpg'.format(lbl1,lbl2), np.clip(im_new,0,1))
    
# make linear blend
    alpha = 0.5
    im1_new = im1_rect - im1_mask
    im2_mask = im2_rect - im2_new
    im_mask = alpha * im1_mask + (1-alpha) * im2_mask
    im_lin = im1_new + im_mask + im2_new
    skio.imsave('{0}_{1}_lin.jpg'.format(lbl1,lbl2), np.clip(im_lin,0,1))

# make gaussian stack blend
    
    mid_pt = int(min_x + int((len(im2[0])-min_x)/2))
    im_gaus = np.zeros(my_shape)
    im_gaus[:,:mid_pt] = 1
    im_mask_3d = np.dstack((im_gaus,im_gaus,im_gaus))
    hybrid = multi_blend(im2_rect, im1_rect, im_mask_3d)
    skio.imsave('{0}_{1}_lap.jpg'.format(lbl1,lbl2), np.clip(hybrid,0,1))
    
    

# rectify image (you need to plot points yourself)

rect_im_1 = rect_image('pictures_raw/unsorted/im3.jpg')

rect_im_2 = rect_image('pictures_raw/unsorted/im4.jpg')


# this code is if you need to input points
# im1_pts, im2_pts = get_points(im1,im2,10)
# im5_pts, im6_pts = get_points(im5, im6, 10)
# im9_pts, im8_pts = get_points(im9, im8, 10)

# make mosiac given points
# east asian library
im1_pts=[(208.53225806451599, 375.06129032258059), (490.72580645161281, 416.86774193548354), (752.01612903225794, 448.22258064516109), (104.01612903225805, 1106.6741935483869), (407.11290322580646, 1138.0290322580645), (668.40322580645159, 1158.9322580645162), (30.854838709677438, 1827.8354838709674), (323.49999999999989, 1838.2870967741933), (616.14516129032256, 1848.7387096774191), (605.69354838709671, 2162.2870967741933)]
im2_pts=[(2476.5322580645161, 563.19032258064499), (2696.016129032258, 563.19032258064499), (2915.5, 521.38387096774159), (2497.4354838709678, 1221.6419354838708), (2706.4677419354839, 1211.190322580645), (2936.4032258064517, 1190.2870967741933), (2486.983870967742, 1848.7387096774191), (2727.3709677419356, 1827.8354838709674), (2978.2096774193546, 1817.3838709677416), (2988.6612903225805, 2110.0290322580645)]
make_mosaic('pictures_raw/first_set/im1.jpg','pictures_raw/first_set/im2.jpg', im1_pts, im2_pts,'im1','im2' )

# inside of the geology building at berkeley
im5_pts = [(4660.8946350312235, 1164.1308895060438), (4677.5385597097593, 1369.4059605413127), (3151.8454641773556, 1846.5318013259916), (3307.1887611770185, 1846.5318013259916), (3523.5597819979771, 1846.5318013259916), (3762.122702390317, 1796.6000272903857), (3817.6024513187681, 2129.478520861092), (3939.6578989613604, 2129.478520861092), (3151.8454641773556, 2612.1523365386156), (3318.2847109627087, 2606.604361645771)]
im6_pts = [(1775.9476907517701, 1180.7748141845791), (1775.9476907517701, 1369.4059605413127), (183.67889650522557, 1741.1202783619347), (388.9539675404942, 1752.2162281476249), (627.51688793283347, 1763.3121779333151), (904.91563257508869, 1724.4763536833993), (943.75145682500442, 2068.4507970397954), (1065.8069044675967, 2068.4507970397954), (150.39104714815494, 2601.0563867529254), (355.66611818342358, 2584.4124620743905)]
make_mosaic('pictures_raw/unsorted/im6.jpg','pictures_raw/unsorted/im5.jpg',im6_pts,im5_pts,'im5','im6')


# pictures at my church building in alameda
im9_pts = [(1451.7277722277722, 1187.176323676324), (2591.0684315684316, 1359.8036963036966), (2577.2582417582421, 2195.3201798201799), (1403.392107892108, 2202.2252747252751)]
im8_pts = [(3438.0806451612907, 1305.2548387096772), (4483.2419354838712, 1367.9645161290323), (4493.6935483870966, 2162.2870967741933), (3438.0806451612907, 2162.2870967741933)]
make_mosaic('pictures_raw/unsorted_2/im9.jpg','pictures_raw/unsorted_2/im8.jpg', im9_pts, im8_pts )

# three image mosaic
im98 = skio.imread('im9_im8_lap.jpg')/255
im7 = skio.imread('pictures_raw/unsorted_2/im7.jpg')/255
im98_pts, im7_pts = get_points(im98, im7,4)
make_mosaic('im9_im8_lap.jpg','pictures_raw/unsorted_2/im7.jpg', im98_pts, im7_pts )







# In[ ]:




