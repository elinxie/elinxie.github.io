
# coding: utf-8

# In[ ]:


# get pictures and then put them in dictionaries with their positions as keys 
from os import listdir
from os.path import isfile, join
import os
import skimage as sk
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


class im_obj:
    def __init__(self, filename):
        self.filename = filename
        self.data_lst = filename[:-4].split("_")
        self.x_index = int(self.data_lst[1])
        self.y_index = int(self.data_lst[2])
        self.x = float(self.data_lst[3])
        self.y = float(self.data_lst[4])
        self.im = skio.imread(filename).astype(int)
    def x_ind(self):
        return self.x_index
    def y_ind(self):
        return self.y_index
    def x_float(self):
        return self.x
    def y_float(self):
        return self.y
    def x_int(self):
        return int(self.x)
    def y_int(self):
        return int(self.y)
    def im_float(self):
        return self.im.astype(float)
    def im_float(self):
        return self.im
    def name(self):
        return self.filename
    def __str__(self):
        return self.filename
    


# In[ ]:


class lightfield:
    def __init__(self, my_path):
        self.onlyfiles = [im_obj(my_path + '/' + f) for f in listdir(my_path) if (isfile(join(my_path, f)) and f != '.DS_Store')]
        self.width = self.onlyfiles[-1].x_ind() + 1
        self.height = self.onlyfiles[-1].y_ind() + 1
        self.im_mat = self.build_matrix()
    def build_matrix(self):
        n = (self.height, self.width)
        new_mat = np.empty(n, dtype=im_obj)
        for o_obj in self.onlyfiles:
            new_mat[o_obj.y_ind()][o_obj.x_ind()] = o_obj
        return new_mat


# In[ ]:


def shift(image, shifts):
    """Shift image, because scipy's shift is uber slow.
    
    99.9% faster than scipy. Tested on 1400x800 image. lol. minus the pizzaz
    """
    assert len(shifts) == len(image.shape), 'Dimensions must match'
    new = np.zeros(image.shape)
    og_selector, target_selector = [], []
    for shift in shifts:
        if shift == 0:
            og_s, target_s = slice(None), slice(None)
        elif shift > 0:
            og_s, target_s = slice(shift, None), slice(None, -shift)
        else:
            og_s, target_s = slice(None, shift), slice(-shift, None)
        og_selector.append(og_s)
        target_selector.append(target_s)
    new[og_selector] = image[target_selector]
    return new


# In[ ]:


def refocus_im(i_obj, m_x, m_y, c):
    return shift(i_obj.im, (int(c*(m_x - i_obj.x)), int(c*(m_y - i_obj.y)), 0))


# In[ ]:


def shift_im_stack(lt_field, c):
    m_x_ind = int((lt_field.width - 1)/2)
    m_y_ind = int((lt_field.height - 1)/2)
    print(m_x_ind)
    m_obj = lt_field.im_mat[m_y_ind][m_x_ind]
    m_x = m_obj.x
    m_y = m_obj.y
    print(m_x)
    print(m_y)
    num_im = float(len(lt_field.onlyfiles))
    new_im_lst = [refocus_im(o_obj, m_x, m_y, c)/num_im for o_obj in lt_field.onlyfiles]
    sum_im = np.copy(new_im_lst[0])
    for new_im in new_im_lst[1:]:
        sum_im += new_im
    return sum_im, new_im_lst


# In[ ]:


def shift_im_mat(im_mat, c):
    m_x_ind = int((len(im_mat[0]) - 1)/2)
    m_y_ind = int((len(im_mat) - 1)/2)
    m_obj = im_mat[m_y_ind][m_x_ind]
    m_x = m_obj.x
    m_y = m_obj.y
    num_im = float(len(im_mat[0]) * len(im_mat))
    new_shape = [int(num_im)]
    if (len(im_mat.shape) > 2):
        for a in im_mat.shape[2:]:
            new_shape.append(a)
    new_shape = tuple(new_shape) 
    im_obj_lst = np.reshape(im_mat, new_shape)
    new_im_lst = [refocus_im(o_obj, m_x, m_y, c)/num_im for o_obj in im_obj_lst]
    sum_im = np.copy(new_im_lst[0])
    for new_im in new_im_lst[1:]:
        sum_im += new_im
    return sum_im, new_im_lst


# In[ ]:


def shift_im_stack(lt_field, c):
    return shift_im_mat(lt_field.im_mat, c)


# In[ ]:


def make_refocus_gif(dir_name, c_start, c_end, step=0.05):
    lt_field = lightfield(dir_name)
    new_dir = dir_name+'/refocused'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for i,c in enumerate(np.arange(c_start, c_end + step, step)):
        new_im, _ = shift_im_stack(lt_field, c)
        fname = dir_name + '/refocused/refocus_' + str(1000 + i) + '.png'
        skio.imsave(fname, new_im/255)
    all_names = dir_name + '/refocused/refocus_*.png' 
    o_name = dir_name + '/refocused/animation_refocus.gif'
    os.system('convert -delay 20 {0} {1}'.format(all_names, o_name))


# In[ ]:


# so what you do is you make a method that takes in a path name and gives you a lightfield
# another function that reduces your matrix from the middle, gets you a lot of images pertaining to that and stacks them
def reduce_mat_middle(im_mat, x, y, n):
    if (len(im_mat) <= 0):
        print('im_mat is empty')
        return None
    if ((x - n < 0 or y - n < 0) or (x + n > len(im_mat[0]) or y + n > len(im_mat))):
        print('n is too large')
        return None
    if (n == 0):
        return np.array([[im_mat[y,x]]])
    else:
        x_start = x - n
        x_end = x + n + 1
        y_start = y - n 
        y_end = y + n + 1
        return im_mat[y_start:y_end, x_start:x_end]


# In[ ]:


def change_aperture(dir_name, start=0, end=8, c=0, lt_field=None):
    if (start < 0):
        print('cannot start from '+str(start))
        return None
    if (lt_field == None):
        lt_field = lightfield(dir_name)
    new_dir = dir_name+'/aperture'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for n in range(start, end + 1):
        m_x_ind = int((lt_field.width - 1)/2)
        m_y_ind = int((lt_field.height - 1)/2)
        print(n)
        new_mat = reduce_mat_middle(lt_field.im_mat, m_x_ind, m_y_ind, n)
        new_im, _ = shift_im_mat(new_mat, c)
        fname = dir_name + '/aperture/ap_' + str(1000 + n) + '.png'
        skio.imsave(fname, np.clip(new_im/255, 0 ,1))
    all_names = dir_name + '/aperture/ap_*.png' 
    o_name = dir_name + '/aperture/animation_aperture.gif'
    os.system('convert -delay 20 {0} {1}'.format(all_names, o_name))
        
    


# In[ ]:


make_refocus_gif('rectified', -0.7, 0.5)


# In[ ]:


change_aperture('rectified', 0, 8, c = 0)


# In[ ]:


change_aperture('treasure', 0, 8, c = 0)


# In[ ]:


make_refocus_gif('treasure', -0.5, 0.5, step=0.1)

