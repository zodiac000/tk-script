import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import pandas as pd
import os
from PIL import Image
from pdb import set_trace



def generate_heatmap(width, height, x_gt, y_gt):
    x = np.zeros((width, height))
    # x[int(x_gt)][int(y_gt)] = 1
    x[int(y_gt), int(x_gt)] = 255
    return x


def show_coordinate(image, coor):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(coor[:, 0], coor[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class CoorToHeatmap(object):
    """Convert coordinates to heatmap
    Args:

        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, coor = sample['image'], sample['coor']
        h, w = image.shape

        coor = coor * [self.output_size / w, self.output_size / h]
        hmap = generate_heatmap(self.output_size, self.output_size, \
                coor[0], coor[1])
        # y = y.reshape(1, h, w)
        hmap = Image.fromarray(np.uint8(hmap))
        # set_trace()
        # return {'image': image, 'hmap': y}
        return hmap 

def heatmap_to_coor(nparray):
    max = np.argmax(nparray)
    y = max // nparray.shape[1]
    x = max % nparray.shape[1]
    # set_trace()

    return x, y

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
    n = 33
    coor_frames = pd.read_csv('saved_dict.csv', header=None)
    img_name = coor_frames.iloc[n, 0]
    coor = coor_frames.iloc[n, 1:].as_matrix()
    coor = coor.astype('float').reshape(-1, 2)
       
    plt.figure()
    show_coordinate(io.imread(os.path.join('./images/', img_name)),
		   coor)
    plt.show()

