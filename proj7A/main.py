
# coding: utf-8

# In[ ]:


# this one is for our fake miniatures
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, inv
import skimage.io as skio
import sys
from skimage import filters


# In[ ]:


# first step is to get DOF line. We do ginput and then we get ourselfs a best fit line through it
# ginput to get points
# do some linalgebra least squares sort of thing to get coeffients a, b of y = ax + b
# or don't make it hard on yourself and just get the y coord

def get_pts(im, num_points=2):
    plt.imshow(im)
    x = plt.ginput(num_points, timeout=200)
    plt.close()
    return x

def get_line_coefficient(x_coords):
    x_vect = [x[0] for x in x_coords]
    y_vect = [x[1] for y in y_coords]
    one_vect = np.ones(x_vect.shape)
    A = np.concatenate(x_vect, one_vect)
    b = y_vect
    coeff = lstsq(A,b)[0]
    return coeff



# In[ ]:


# let's create our mask here
# we run the gaussian here
def get_mask_y_coord(im_size, y_top, y_bottom, vert=False, mask_val = 1.0):
    mask = np.zeros(im_size)
    if (vert):
        mask[:,max(int(y_top),0): min(int(y_bottom),len(mask))] = mask_val
    else:
        mask[max(int(y_top),0): min(int(y_bottom),len(mask))] = mask_val
    return mask
    
def gaus_picture(im, n, sigma=1):
    new_im = im
    for i in range(0,n):
        new_im = filters.gaussian(new_im, sigma)
    return new_im


# In[ ]:


# then we can take the picture again and run the gaussian x amount of times 


# In[ ]:


#we then use the blending of the three pictures, original, blurred and the mask like the function below
def makeLS(LA, LB, GR):
    LS = np.zeros((len(LA),len(LA[0]),1))
    LS += GR*LA
    LS += (1-GR)*LB
    return np.clip(LS,0,1)


# In[122]:


def blend_channel(im_ch, coords, margin=200, n = 5, sigma=2,vert=False):
    if (vert):
        y_top = coords[0][0]
        y_bottom = coords[1][0]
    else:
        y_top = coords[0][1]
        y_bottom = coords[1][1]
    o_mask = get_mask_y_coord(im_ch.shape, y_top, y_bottom,vert=vert)
    gaus_mask = np.clip(gaus_picture(o_mask, n, sigma),0,1.0)
    gaus_im = np.clip(gaus_picture(im_ch, n, sigma), 0 ,1.0 )
    return makeLS(im_ch, gaus_im, gaus_mask)


# In[126]:


# let's write the main function
# but I think we have to split the color channels 
def fake_mini(imname,sigma=3, vert=False):
    im = (skio.imread(imname)/255.0).astype(np.float64)
    g_im = (skio.imread(imname)).astype(np.float64)
    print(im)
    coords = get_pts(im,2)
    im_channels = np.dsplit(im,3)
    fake_ch = [blend_channel(i,coords,sigma=sigma,vert=vert) for i in im_channels]
    new_im = np.dstack(fake_ch)
    fname = 'fake_'+ imname
    plt.imshow(new_im)
    skio.imsave(fname, new_im)


# In[127]:


def main(argv):
    if len(argv) < 2:
        print('please give an image to input')
        return
    imname = argv[1]
    sigma = 3
    vert = False
    if len(argv) > 2:
        sigma = float(argv[2])
    if len(argv) > 3:
        vert = bool(argv[3])
    fake_mini(imname, sigma, vert)


if __name__ == "__main__":
    main(sys.argv)
