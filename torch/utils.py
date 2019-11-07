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
        y = generate_heatmap(self.output_size, self.output_size, \
                coor[0, 0], coor[0, 1])
        # y = y.reshape(1, h, w)
        y = Image.fromarray(np.uint8(y))
        return {'image': image, 'coor': y}

def heatmap_to_coor(nparray):
    max = np.argmax(nparray)
    # set_trace()
    y = max // nparray.shape[1]
    x = max % nparray.shape[1]

    return x, y

def accuracy(outputs, labels):
    coor_outputs = []
    coor_labels = []
    for out in outputs:
        x, y = heatmap_to_coor(out)
        coor_outputs.append((x, y))
    for label in labels:
        x, y = heatmap_to_coor(label)
        coor_labels.append((x, y))

    acc_x = 0
    acc_y = 0
    for idx, output in enumerate(coor_outputs):
        acc_x += (1 - abs(output[0] - coor_labels[idx][0]) / 224)
        acc_y += (1 - abs(output[1] - coor_labels[idx][1]) / 224)
        # set_trace()

    return acc_x / len(outputs), acc_y / len(outputs)

# https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Pad
class Padding(object):
    """Adding two more empty channels to the original image.
    Args:


    """
    def __init__(self):
        pass

    def __call__(self, sample):
        image, coor = sample['image'], sample['coor']
        h, w = image.shape[:2]
        image = image.reshape(1, h, w)
        # zeros = np.zeros((2, h, w))
        # image = np.concatenate((image, zeros), axis=0)

        return {'image': image, 'coor': coor}



# https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Resize
# https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Resize
class Rescale(object):
    """Rescale the image in a sample to a given size.

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

        h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
            # if h > w:
                # new_h, new_w = self.output_size * h / w, self.output_size
            # else:
                # new_h, new_w = self.output_size, self.output_size * w / h
        # else:
            # new_h, new_w = self.output_size

        # new_h, new_w = int(new_h), int(new_w)

        if isinstance(self.output_size, int):
            img = transform.resize(image, (self.output_size, self.output_size))

        # h and w are swapped for coor because for images,
        # x and y axes are axis 1 and 0 respectively
        coor = coor * [self.output_size / w, self.output_size / h]

        return {'image': img, 'coor': coor}

# https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, coor = sample['image'], sample['coor']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'coor': torch.from_numpy(coor).float()}

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

