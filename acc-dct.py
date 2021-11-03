import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from skimage import color, io, data
from skimage.util import view_as_blocks
from PIL import Image, ImageOps
#from skimage.transform import rescale, resize
from scipy.fftpack import dct
from transform import *
from skimage.io import imread, imshow
import os
from scipy.spatial import distance
from skimage import color, io, data
from skimage.util import view_as_blocks
from skimage.transform import rescale, resize
from scipy.fftpack import dct
import scipy
import glob
from scipy.fftpack import fft, dct

def dct_calculate(split,ctr,k):

    a=scipy.fftpack.dct(split, type=2, n = None, axis=- 1, norm='ortho', overwrite_x=False)

    amax, amin = a.max(), a.min()
    a_norm = (a - amin)/(amax - amin)
    d=np.mean(a_norm,axis=0)
    #print(d)
    #print(d.shape)#50
    column_zero = [row[0] for row in d]
    #print(column_zero)
    x= np.array(column_zero)
    #PRINT THE MATRIX after normalization
    #print(x)
    average_of_each_block=sum(x)/len(x)
    #print(average_of_each_block)
    #combine all values
    arr[count][k-1]=average_of_each_block

def imgcrop(input,count):
    filename, file_extension = os.path.splitext(input)
    im = Image.open(input)
    #imgwidth, imgheight = im.size
    #height = imgheight // rows
    #width = imgwidth // columns
    ctr=count

    affine=im
    for k in range(1,9):
        if k == 1:
            dct_calculate(affine,ctr,k)
            try:
                affine.save("images/" + filename + "-" + str(i) + "-" + str(j) + "-" + str(k) + file_extension)
            except:
                pass

        if k == 2 :
            im_mirror = ImageOps.mirror(affine)
            dct_calculate(im_mirror,ctr,k)
            try:
                im_mirror.save("images/" + filename + "-" + str(i) + "-" + str(j) + "-" + str(k) + file_extension,quality=95)
            except:
                pass

        if k == 3 :
            angle = 90
            affine = affine.rotate(angle, expand=True)
            dct_calculate(affine,ctr,k)
            try:
                affine.save("images/" + filename + "-" + str(i) + "-" + str(j) + "-" + str(k) + file_extension)
            except:
                pass

        if k == 4 :
            im_mirror = ImageOps.mirror(affine)
            dct_calculate(im_mirror,ctr,k)
            try:
                im_mirror.save("images/" + filename + "-" + str(i) + "-" + str(j) + "-" + str(k) + file_extension,quality=95)
            except:
                pass

        if k == 5 :
            angle = 90
            affine = affine.rotate(angle, expand=True)
            dct_calculate(affine,ctr,k)
            try:
                affine.save("images/" + filename + "-" + str(i) + "-" + str(j) + "-" + str(k) + file_extension)
            except:
                pass

        if k == 6 :
            im_mirror = ImageOps.mirror(affine)
            dct_calculate(im_mirror,ctr,k)
            try:
                im_mirror.save("images/" + filename + "-" + str(i) + "-" + str(j) + "-" + str(k) + file_extension,quality=95)
            except:
                pass

        if k == 7 :
            angle = 90
            affine = affine.rotate(angle, expand=True)
            dct_calculate(affine,ctr,k)
            try:
                affine.save("images/" + filename + "-" + str(i) + "-" + str(j) + "-" + str(k) + file_extension)
            except:
                pass

        if k == 8 :
            im_mirror = ImageOps.mirror(affine)
            dct_calculate(im_mirror,ctr,k)
            try:
                im_mirror.save("images/" + filename + "-" + str(i) + "-" + str(j) + "-" + str(k) + file_extension,quality=95)
            except:
                pass

f='001_F.png'
image = Image.open(f)
greyscale_image = image.convert('L')
#greyscale_image.show()

greyscale_image.save("greyscale/" + '001_F.png')
rimg = greyscale_image.resize((128, 128))
rimg.save("greyscale/resize/" + '001_F.png')
#rimg.show()

#path_to_rimg = "/greyscale/resize/"
rimg = cv2.imread('greyscale/resize/001_F.png')
rimg_h, rimg_w, _ = rimg.shape
split_width = 50
split_height = 50

#change overlap accordingly to overlap more or less!
def start_points(size, split_size, overlap=50):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

X_points = start_points(rimg_w, split_width, 0.5)
Y_points = start_points(rimg_h, split_height, 0.5)

count = 0
name = 'splitted'
frmt = 'jpeg'
rows, cols = (25, 8)
arr = [[0 for i in range(cols)] for j in range(rows)]
#for i in arr:
#    print(i)

for i in Y_points:
    for j in X_points:
        #print(count)
        split_image = rimg[i:i+split_height, j:j+split_width]
        cv2.imwrite('{}_{}.{}'.format(name, count, frmt), split_image)
        filename=name+'_'+str(count)+'.'+frmt
        imgcrop(filename,count)
        count += 1

for element in arr:
    print(element)
