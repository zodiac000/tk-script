import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import pandas as pd
import os
from PIL import Image
from numpy import unravel_index
from math import exp
from random import random

from pdb import set_trace

def grid_hmap(shape, x_gt, y_gt):
    mask = np.zeros((shape, shape))
    # mask[int(x_gt)][int(y_gt)] = 1
    # set_trace()
    if x_gt > shape or y_gt > shape:
        print('invalid coordinates labels ---- {}, {}'.format(x_gt, y_gt))


    if x_gt > -1 and y_gt > -1:
        size = 13
        # mask[int(y_gt), int(x_gt)] = 255
        #9x9
        # for i in range(-4, 5):
            # for j in range(-4,5):
                # if abs(i) < 1 and abs(j) < 1:
                    # mask[int(y_gt)+i, int(x_gt)+j] = 255
                # else:
                    # mask[int(y_gt)+i, int(x_gt)+j] = 255 - (i**2 + j**2)
        #13x13
        # for i in range(-6, 7):
            # for j in range(-6,7):
                # if abs(i) < 1 and abs(j) < 1:
                    # mask[int(y_gt)+i, int(x_gt)+j] = 255
                # else:
                    # mask[int(y_gt)+i, int(x_gt)+j] = 255 - (i**2 + j**2)
        # 17x17
        # for i in range(-8, 9):
            # for j in range(-10,11):
                # if abs(i) < 2 and abs(j) < 2:
                    # mask[int(y_gt)+i, int(x_gt)+j] = 255
                # else:
                    # mask[int(y_gt)+i, int(x_gt)+j] = 255 - (i**2 + j**2)
        # 21x21
        for i in range(-int(size/2), int(size/2)+1):
            for j in range(-int(size/2), int(size/2)+1):
                if abs(i) < 1 and abs(j) < 1:
                    mask[int(y_gt)+i, int(x_gt)+j] = 255
                else:
                    try:
                        mask[int(y_gt)+i, int(x_gt)+j] = 255 - (i**2 + j**2)
                        # mask[int(y_gt)+i, int(x_gt)+j] = 255 - (i*5 + j*5)
                    except:
                        print(x_gt)
                        print(y_gt)

    return mask


def gaussion_hmap(x, y, shape=224):
    # Probability as a function of distance from the center derived
    # from a gaussian distribution with mean = 0 and stdv = 1
    scaledGaussian = lambda x : exp(-(1/2)*(x**2))
    
    isotropicMask = np.zeros((shape,shape))
    # scalor = random()*3+1
    scalor = 0.1
    boundary = 6
    x = int(x)
    y = int(y)
    
    for j in range(max(0, int(x-boundary)), min(int(x+boundary+1), int(shape))):
        for i in range(max(0, int(y-boundary)), min(int(y+boundary+1), int(shape))):
            # find euclidian distance from center of image (shape/2,shape/2)
            # and scale it to range of 0 to 2.5 as scaled Gaussian
            # returns highest probability for x=0 and approximately
            # zero probability for x > 2.5
            distanceFromLabel = np.linalg.norm(np.array([i-y,j-x]))
            distanceFromLabel = scalor * distanceFromLabel
            scaledGaussianProb = scaledGaussian(distanceFromLabel)
            isotropicMask[i,j] = np.clip(scaledGaussianProb*255,0,255)
            


    return np.round(isotropicMask)


def heatmap_to_coor(nparray):
    max = np.argmax(nparray)
    y = max // nparray.shape[1]
    x = max % nparray.shape[1]
    # y, x = unravel_index(nparray.argmax(), nparray.shape)
    return x, y

def get_total_confidence(nparray):
    x, y = heatmap_to_coor(nparray)
    sum = []
    n_1 = 0
    n_2 = 0
    n_3 = 0
    n_4 = 0
    n_5 = 0

    for i in range(y-3, y+4):
        for j in range(x-3, x+4):
            if i in range(0, 223) and j in range(0, 223):
                sum.append(nparray[i, j])
            else:
                sum.append(0)

    for i in sum:
        if i < 1e-1:
            n_1 += 1
        if i < 1e-2:
            n_2 += 1
        if i < 1e-3:
            n_3 += 1
        if i < 1e-4:
            n_4 += 1
        if i < 1e-5:
            n_5 += 1

    return [n_1, n_2, n_3, n_4, n_5]

class CoorToHeatmap(object):
    """Convert coordinates to heatmap
    Args:

        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, input_size=(1024, 1280), output_size=224):
        assert isinstance(output_size, (int, tuple))
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, coor):
        h, w = self.input_size

        coor = coor * [self.output_size / w, self.output_size / h]
        coor = np.round(coor)
        
        # hmap = grid_hmap(self.output_size, coor[0], coor[1])
        hmap = gaussion_hmap(coor[0], coor[1])


        # print(unravel_index(hmap.argmax(), hmap.shape))
        # print(hmap.max())
        # i, j = unravel_index(hmap.argmax(), hmap.shape)
        # print(hmap[i-10:i+10, j-10:j+10])
        # set_trace()


        # hmap = Image.fromarray(np.uint8(hmap))
        return hmap 

def generate_heatmap2(w, h, x_gt, y_gt):
    x_range = np.arange(start=0, stop=w, dtype=int)
    y_range = np.arange(start=0, stop=h, dtype=int)
    xx, yy = np.meshgrid(x_range, y_range)
    d2 = (xx - int(x_gt))**2 + (yy - int(y_gt))**2
    sigma = 2
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = np.exp(-exponent)
    return heatmap

def accuracy_sum(outputs, labels):
    coor_outputs = []
    coor_labels = []
    list_acc_x = []
    list_acc_y = []
    for out in outputs:
        x, y = heatmap_to_coor(out)
        coor_outputs.append((x / 224 * 1280, y / 224 * 1024))
    # for label in labels:
        # x, y = heatmap_to_coor(label)
        # coor_labels.append((x, y))

    sum_acc_x = 0
    sum_acc_y = 0
    for idx, output in enumerate(coor_outputs):
        acc_x = (1 - abs(output[0] - labels[idx][0]) / 1280)
        acc_y = (1 - abs(output[1] - labels[idx][1]) / 1024)
        sum_acc_x += acc_x
        sum_acc_y += acc_y
        list_acc_x.append(acc_x)
        list_acc_y.append(acc_y)
        # set_trace()

    # return sum_acc_x / len(outputs), sum_acc_y / len(outputs)
    return sum_acc_x, sum_acc_y, list_acc_x, list_acc_y

def spike(hmap):
    x, y = hmap.squeeze().max(1)[0].max(0)[1].item(), hmap.squeeze().max(0)[0].max(0)[1].item()
    new_hmap = torch.zeros(hmap.shape)
    new_hmap[0, x, y] = 1
    return new_hmap

def crop(image, w_center, h_center, coor, scale):
    h_image, w_image = image.shape
    image_np = image.numpy()
    w_center = int(w_center / 224 * w_image)
    h_center = int(h_center / 224 * h_image)
    if w_center - (scale/2) < 0:
        w_left = 0
        w_right = scale
        left_margin = 0
    elif w_center + (scale/2) > w_image:
        w_left = w_image - scale
        w_right = w_image
        left_margin = w_left
    else:
        w_left = w_center - int(scale/2)
        w_right = w_center + int(scale/2)
        left_margin = w_left

    if h_center - (scale/2) < 0:
        h_top = 0
        h_bottom = scale
        top_margin = 0
    elif h_center + (scale/2) > h_image:
        h_top = h_image -scale
        h_bottom = h_image
        top_margin = h_top
    else:
        h_top = h_center - int(scale/2)
        h_bottom = h_center + int(scale/2)
        top_margin = h_top
    hmap = torch.zeros(224, 224)
    if h_top <= coor[0] < h_bottom and \
        w_left <= coor[1] < w_right:
            hmap[int((coor[0] - top_margin)  / (scale/224)), int((coor[1] - left_margin) / (scale/224))] = 1
    
    return image[h_top:h_bottom, w_left:w_right], hmap


# def crop_112(image, w_center, h_center, coor):
    # h_image, w_image = image.shape
    # image_np = image.numpy()
    # w_center = int(w_center / 224 * w_image)
    # h_center = int(h_center / 224 * h_image)
    # # coor = coor * [self.output_size / w, self.output_size / h]
    # if w_center - 56 < 0:
        # w_left = 0
        # w_right = 112
        # left_margin = 0
    # elif w_center + 56 > w_image:
        # w_left = w_image - 112
        # w_right = w_image
        # left_margin = w_left
    # else:
        # w_left = w_center - 56
        # w_right = w_center + 56
        # left_margin = w_left

    # if h_center - 56 < 0:
        # h_top = 0
        # h_bottom = 112
        # top_margin = 0
    # elif h_center + 56 > h_image:
        # h_top = h_image -112
        # h_bottom = h_image
        # top_margin = h_top
    # else:
        # h_top = h_center - 56
        # h_bottom = h_center + 56
        # top_margin = h_top
    # hmap = torch.zeros(224, 224)
    # if coor[0] >= h_top and coor[0] < h_bottom and \
        # coor[1] >= w_left and coor[1] < w_right:
            # hmap[(coor[0] - top_margin) * 2, (coor[1] - left_margin) * 2] = 1
    # image_112 = image[h_top:h_bottom, w_left:w_right]

    # return image_112, hmap


if __name__ == "__main__":

    hmap = gaussion_hmap(100,200)
    
    set_trace()
