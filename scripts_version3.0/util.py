import skimage.io
import skimage.transform
from PIL import ImageFile
import os
import ipdb
import scipy.misc as misc
import numpy as np
import math

#def load_image( path, height=128, width=128 ):
def load_image( path,net_size=128, pre_height=128, pre_width=128, height=128, width=128 ):
    pre_height=net_size
    pre_width=net_size
    height=net_size
    width=net_size
    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    img /= 255.

#    img = math.pow(10/(img+1e-5),2.3)
    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )

    #rand_y = np.random.randint(0, pre_height - height)
    #rand_x = np.random.randint(0, pre_width - width)

    #resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]
    #resized_img[resized_img>0.7]=1
    #resized_img[resized_img<0.7]=0
    return resized_img
    #return (resized_img * 2)-1 #(resized_img - 127.5)/127.5

def load_chain( path,net_size=128, pre_length=128 ):
    pre_length=net_size
    try:
        chain = np.loadtxt( path )
    except:
        return None
    #temp=chain.split()
    chain_1=chain[0]
    chain_2=chain[1]
    a=chain_1+chain_2    
    b=chain_1*pre_length
    d=b//a
    c=int(d)
    return c

def crop_random(image_ori, width=64,height=64, x=None, y=None, overlap=7):
    if image_ori is None:
        return None
    random_y = np.random.randint(overlap,height-overlap) if x is None else x
    random_x = np.random.randint(overlap,width-overlap) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()
    crop = crop[random_y:random_y+height, random_x:random_x+width]
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 0] = 2*117. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 1] = 2*104. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 2] = 2*123. / 255. - 1.

    return image, crop, random_x, random_y

def crop_interaction(image_ori, c,net_size=128):
    if image_ori is None: 
        return None
#    random_y = np.random.randint(overlap,height-overlap) if x is None else x
#    random_x = np.random.randint(overlap,width-overlap) if y is None else y
    x=c
    y=c
    hid_size=net_size/2
    image = image_ori.copy()
    image_in=image_ori.copy()
    crop_temp = image_ori.copy()
    crop_temp2 = crop_temp[0:y, x:net_size]
    image[0:y, x:net_size, 0] = 1.0
    image[0:y, x:net_size, 1] = 1.0
    image[0:y, x:net_size, 2] = 1.0
#    crop_1=misc.imresize(crop_temp2[:,:,0],[hid_size,hid_size],interp='nearest')
 #   crop_2=misc.imresize(crop_temp2[:,:,1],[hid_size,hid_size],interp='nearest')
  #  crop_3=misc.imresize(crop_temp2[:,:,2],[hid_size,hid_size],interp='nearest')
#    crop_temp3=crop_temp
#    crop=crop_temp[0:hid_size,hid_size:net_size]
    crop=np.zeros_like(image_ori)
    crop[0:y, x:net_size,0]=crop_temp2[:,:,0]
    crop[0:y, x:net_size,1]=crop_temp2[:,:,1]
    crop[0:y, x:net_size,2]=crop_temp2[:,:,2]
    return image,image_in, crop,x,y

def crop_interaction_rotate(image_ori_t, c,net_size=128):
    if image_ori_t is None:
        return None
#    random_y = np.random.randint(overlap,height-overlap) if x is None else x
#    random_x = np.random.randint(overlap,width-overlap) if y is None else y
    x=c
    y=c
    z=net_size-y
    hid_size=net_size/2
    image_ori_1=image_ori_t[0:y,0:y]
    image_ori_2=image_ori_t[0:y,y:net_size]
#    image_ori_3=image_ori_t[y:net_size,y:net_size]
    image_ori_1=image_ori_1[:,::-1,:]
    image_ori_1=image_ori_1[::-1,:,:]
    image_ori_2=image_ori_2[:,::-1,:]
    image_ori=image_ori_t.copy()
    image_ori[0:y, 0:y, 0]=image_ori_1[0:y, 0:y, 0]
    image_ori[0:y, 0:y, 1]=image_ori_1[0:y, 0:y, 1]
    image_ori[0:y, 0:y, 2]=image_ori_1[0:y, 0:y, 2]
    image_ori[0:y, y:net_size, 0]=image_ori_2[0:y, 0:z, 0]
    image_ori[0:y, y:net_size, 1]=image_ori_2[0:y, 0:z, 1]
    image_ori[0:y, y:net_size, 2]=image_ori_2[0:y, 0:z, 2]
    image = image_ori.copy()
    image_in=image_ori.copy()
    crop_temp = image_ori.copy()
    crop_temp2 = crop_temp[0:y, x:net_size]
    image[0:y, x:net_size, 0] = 0.6
    image[0:y, x:net_size, 1] = 0.6
    image[0:y, x:net_size, 2] = 0.6
   # crop_1=misc.imresize(crop_temp2[:,:,0],[hid_size,hid_size],interp='nearest')
   # crop_2=misc.imresize(crop_temp2[:,:,1],[hid_size,hid_size],interp='nearest')
   # crop_3=misc.imresize(crop_temp2[:,:,2],[hid_size,hid_size],interp='nearest')
#    crop_temp3=crop_temp
   # crop=crop_temp[0:hid_size,hid_size:net_size]
   # crop[:,:,0]=crop_1
   # crop[:,:,1]=crop_2
   # crop[:,:,2]=crop_3
    crop=np.zeros_like(image_ori)
    crop[0:y, x:net_size,0]=crop_temp2[:,:,0]
    crop[0:y, x:net_size,1]=crop_temp2[:,:,1]
    crop[0:y, x:net_size,2]=crop_temp2[:,:,2]
    return image,image_in, crop,x,y

def crop_interaction_swap(image_ori_t, c,net_size=128):
    if image_ori_t is None:
        return None
#    random_y = np.random.randint(overlap,height-overlap) if x is None else x
#    random_x = np.random.randint(overlap,width-overlap) if y is None else y
    x=c
    y=c
    z=net_size-x
    hid_size=net_size/2
    image_ori_1=image_ori_t[0:y,0:y]
    image_ori_2=image_ori_t[0:y,y:net_size]
    image_ori_3=image_ori_t[y:net_size,y:net_size]
    image_ori_2_=np.transpose(image_ori_2,(1,0,2))
#    image_ori_1=image_ori_1[:,::-1,:]
 #   image_ori_2=image_ori_2[:,::-1,:]
    image_ori=image_ori_t.copy()
    image_ori[0:z,0:z, 0]=image_ori_3[0:z,0:z, 0]
    image_ori[0:z,0:z,1]=image_ori_3[0:z,0:z,1]
    image_ori[0:z,0:z,2]=image_ori_3[0:z,0:z, 2]
    image_ori[0:z, z:net_size, 0]=image_ori_2_[0:z, 0:x, 0]
    image_ori[0:z, z:net_size, 1]=image_ori_2_[0:z, 0:x, 1]
    image_ori[0:z, z:net_size, 2]=image_ori_2_[0:z, 0:x, 2]
    image_ori[z:net_size, z:net_size, 0]=image_ori_1[0:y, 0:x, 0]
    image_ori[z:net_size, z:net_size, 1]=image_ori_1[0:y, 0:x, 1]
    image_ori[z:net_size, z:net_size, 2]=image_ori_1[0:y, 0:x, 2]
    image_ori[z:net_size,0:z, 0]=0
    image_ori[z:net_size,0:z, 1]=0
    image_ori[z:net_size,0:z, 2]=0
    image = image_ori.copy()
    image_in=image_ori.copy()
    crop_temp = image_ori.copy()
    crop_temp2 = crop_temp[0:z, z:net_size]
    image[0:z, z:net_size, 0] = 0.6
    image[0:z, z:net_size, 1] = 0.6
    image[0:z, z:net_size, 2] = 0.6
#    crop_1=misc.imresize(crop_temp2[:,:,0],[hid_size,hid_size],interp='nearest')
#    crop_2=misc.imresize(crop_temp2[:,:,1],[hid_size,hid_size],interp='nearest')
#    crop_3=misc.imresize(crop_temp2[:,:,2],[hid_size,hid_size],interp='nearest')
#    crop_temp3=crop_temp
 #   crop=crop_temp[0:hid_size,hid_size:net_size]
 #   crop[:,:,0]=crop_1
 #   crop[:,:,1]=crop_2
 #   crop[:,:,2]=crop_3
    crop=np.zeros_like(image_ori)
    crop[0:z, z:net_size,0]=crop_temp2[:,:,0]
    crop[0:z, z:net_size,1]=crop_temp2[:,:,1]
    crop[0:z, z:net_size,2]=crop_temp2[:,:,2]
    x=z
    y=z
    return image,image_in, crop,x,y




#p1 = "/extend2/data_version_2.0/datasets/3dcomplex_homodimers/contact/train_images/"
#p2 = "/extend2/data_version_2.0/datasets/3dcomplex_homodimers/contact/train_chains/"
#image_ori = load_image(p1+"9rub_1.jpg")
#chain_ori = load_chain(p2+"9rub_1.txt")
#print(chain_ori)
#print(image_ori)
#c = crop_interaction(image_ori,chain_ori)
#print(c)
