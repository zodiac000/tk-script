import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
from skimage import io, transform
from torchvision import utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import CoorToHeatmap
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.transforms.functional import crop
from PIL import Image
import matplotlib.pyplot as plt
from torch import cat
from random import randint
from pdb import set_trace

# Helper function to show a batch


def show_coor_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, coor_batch = \
        sample_batched['image'], sample_batched['coor_bc']
    # batch_size = len(images_batch)
    # im_size = images_batch.size(2)
    # grid_border_size = 2

    grid_image = utils.make_grid(images_batch)
    grid_coor = utils.make_grid(coor_batch)
    grid = np.concatenate((grid_image.numpy(), grid_coor.numpy()), axis=1)
    plt.imshow(grid.transpose((1, 2, 0)))

# cutout from image


def cutout_image(image_np, coors, cut_size=40, random=False, scalar=4):
    half_size = cut_size // 2
    x, y = np.clip(coors, half_size, 1024-half_size)

    if random:
        for i in range(y-half_size, y+half_size):
            for j in range(x-half_size, x+half_size*scalar):
                image_np[i, j] = np.random.randint(0, 255)
    else:
        image_np[y-half_size:y+half_size,
                 x-half_size:x+half_size*scalar] = 0


def crop_image(image_np, coors, crop_size=224):
    image_pil = Image.fromarray(image_np)
    top_left = [max(0, x-crop_size//2) for x in coors[::-1]]
    image_crop = crop(image_pil, *top_left, crop_size, crop_size)

    return image_crop

class WeldingDatasetToTensor(Dataset):
    """Welding dataset."""

    def __init__(self, data_root, csv_file, root_dir='.',
                 dist_lower_bound=10):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.dist_lower_bound = dist_lower_bound
        self.data_root = data_root

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # print("getitem_index: {}".format(str(idx)))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, self.data_root,
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        coor_1 = self.all_data.iloc[idx, 1:3]
        coor_1 = np.array(coor_1).astype(int)

        hmap = CoorToHeatmap(image_np.shape, 224)(coor_1)
        hmap = torch.from_numpy(hmap).view(1, 224, 224)
        # hmap = nn.Softmax(2)(hmap.view(-1, 1, 224*224)).view(1, 224, 224)
        input_transform = Compose([Resize((224, 224)), ToTensor()])

        image = input_transform(image_pil)

        sample = {
            'image': image,
            'hmap': hmap,
            'img_name': image_name,
            'coor_1': coor_1,
            'origin_img': image_np,
        }

        return sample


class FakeData(Dataset):
    """Welding dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # print("getitem_index: {}".format(str(idx)))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        # img_name = os.path.join(self.root_dir + 'fail_images/',
        img_name = os.path.join(self.root_dir + 'all_images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        bc_pred_acc = self.all_data.iloc[idx, 1:]
        coor_bc = np.array(bc_pred_acc[:2]).astype(int)
        coor_bc_fake = np.array(bc_pred_acc[2:]).astype(int)

        hmap = CoorToHeatmap(image_np.shape, 224)(coor_bc)
        hmap = torch.from_numpy(hmap)
        hmap = nn.Softmax(2)(hmap.view(-1, 1, 224*224)).view(1, 224, 224)

        hmap_fake = CoorToHeatmap(image_np.shape, 224)(coor_bc_fake)
        hmap_fake = torch.from_numpy(hmap_fake)
        hmap_fake = nn.Softmax(2)(
            hmap_fake.view(-1, 1, 224*224)).view(1, 224, 224)
        input_transform = Compose([Resize((224, 224)), ToTensor()])

        image = input_transform(image_pil)

        sample = {'image': image,
                  'hmap': hmap,
                  'img_name': image_name,
                  'coor_bc': coor_bc,
                  'origin_img': image_np,
                  'coor_bc_fake': coor_bc_fake,
                  'hmap_fake': hmap_fake,

                  # 'pred_bc': pred_bc,\
                  # 'acc_xy': acc_xy,\

                  # 'class_real': class_real, \
                  }
        # 'class_real': class_real, 'dx_dy':dx_dy}

        return sample


class CutoutDataset(Dataset):
    """Cutout image for classification."""

    def __init__(self, csv_file, root_dir='.', dist_lower_bound=10,
                 n_random_cutout=3):
        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.dist_lower_bound = dist_lower_bound
        self.n_random_cutout = n_random_cutout

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, 'all_images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        coors = np.array(self.all_data.iloc[idx, 1:3]).astype(int)
        # top_left = [max(0, x-cut_size//2) for x in coors[::-1]]
        # image_crop = crop(image_pil, *top_left, cut_size, cut_size)
        # image_crop = ToTensor()(image_crop)
        coor_cutout = np.copy(image_np)
        cutout_image(coor_cutout, coors)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        coor_cutout = input_transform(Image.fromarray(coor_cutout))

        image = input_transform(image_pil)

        random_cutouts = []
        for i in range(self.n_random_cutout*4):
            while(True):
                random_delta = np.asarray(
                    [randint(-300, 300), randint(-300, 300)])
                random_coors = np.clip(coors + random_delta, 50, 950)
                dist = np.sum((random_coors - coors) ** 2) ** 0.5
                if (dist > self.dist_lower_bound):
                    image_random_cutout = np.copy(image_np)
                    cutout_image(image_random_cutout, random_coors)
                    image_random_cutout = input_transform(
                        Image.fromarray(image_random_cutout))
                    random_cutouts.append(image_random_cutout)
                    break
        random_cutouts = torch.stack(random_cutouts)

        # generating cutouts within certain Euclidean distance
        random_center_cutouts = []
        for i in range(self.n_random_cutout):
            while(True):
                random_delta = np.asarray([randint(-10, 10), randint(-10, 10)])
                random_coors = coors + random_delta
                dist = np.sum((random_coors - coors) ** 2) ** 0.5
                # print(random_coors, dist)
                if (dist < self.dist_lower_bound):
                    image_random_cutout = np.copy(image_np)
                    cutout_image(image_random_cutout, random_coors)
                    image_random_cutout = input_transform(
                        Image.fromarray(image_random_cutout))
                    random_center_cutouts.append(image_random_cutout)
                    break
        random_center_cutouts = torch.stack(random_center_cutouts)

        sample = {
            'image': image,
            'coor_cutout': coor_cutout,
            'random_cutouts': random_cutouts,
            'random_center_cutouts': random_center_cutouts,
        }

        return sample


class CutoutDataset_pred(Dataset):
    def __init__(self, csv_file, root_dir='.'):
        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, 'all_images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        coor_label = np.array(self.all_data.iloc[idx, 1:3]).astype(int)
        coor_pred = np.array(self.all_data.iloc[idx, 3:]).astype(int)
        coor_cutout = np.copy(image_np)
        cutout_image(coor_cutout, coor_pred)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        coor_cutout = input_transform(Image.fromarray(coor_cutout))
        image = input_transform(image_pil)

        sample = {
            'image_name': image_name,
            'coor_label': coor_label,
            'coor_pred': coor_pred,
            'coor_cutout': coor_cutout,
        }

        return sample


class InvalidDataset(Dataset):
    """Invalid image for classification."""

    def __init__(self, csv_file, root_dir='.', n_random_cutout=2):
        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.n_random_cutout = n_random_cutout

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, 'all_images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        image = input_transform(image_pil)
        random_cutouts = []
        for i in range(self.n_random_cutout):
            random_coors = np.asarray([randint(100, 800), randint(100, 800)])
            image_random_cutout = np.copy(image_np)
            cutout_image(image_random_cutout, random_coors)
            image_random_cutout = input_transform(
                Image.fromarray(image_random_cutout))
            random_cutouts.append(image_random_cutout)

        random_cutouts = torch.stack(random_cutouts)

        sample = {
            'image': image,
            'random_cutouts': random_cutouts,
        }
        return sample


class CropDataset(Dataset):
    """Cutout image for classification."""

    def __init__(self, csv_file, root_dir='.', dist_lower_bound=10,
                 n_random_crop=1):
        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.dist_lower_bound = dist_lower_bound
        self.n_random_crop = n_random_crop

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, 'all_images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        coors = np.array(self.all_data.iloc[idx, 1:3]).astype(int)
        coor_crop = np.copy(image_np)
        coor_crop = crop_image(coor_crop, coors)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        coor_crop = input_transform(coor_crop)

        image = input_transform(image_pil)

        random_crops = []
        for i in range(self.n_random_crop):
            while(True):
                random_delta = np.asarray(
                    [randint(-300, 300), randint(-300, 300)])
                random_coors = np.clip(coors + random_delta, 50, 950)
                dist = np.sum((random_coors - coors) ** 2) ** 0.5
                if (dist > self.dist_lower_bound):
                    image_random_crop = np.copy(image_np)
                    image_random_crop = crop_image(image_random_crop, random_coors)
                    image_random_crop = input_transform(image_random_crop)
                    random_crops.append(image_random_crop)
                    break
        random_crops = torch.stack(random_crops)

        # # generating cutouts within certain Euclidean distance
        # random_center_cutouts = []
        # for i in range(self.n_random_crop):
            # while(True):
                # random_delta = np.asarray([randint(-10, 10), randint(-10, 10)])
                # random_coors = coors + random_delta
                # dist = np.sum((random_coors - coors) ** 2) ** 0.5
                # # print(random_coors, dist)
                # if (dist < self.dist_lower_bound):
                    # image_random_cutout = np.copy(image_np)
                    # cutout_image(image_random_cutout, random_coors)
                    # image_random_cutout = input_transform(
                        # Image.fromarray(image_random_cutout))
                    # random_center_cutouts.append(image_random_cutout)
                    # break
        # random_center_cutouts = torch.stack(random_center_cutouts)

        sample = {
            'image': image,
            'coor_crop': coor_crop,
            'random_crop': random_crops,
            # 'random_center_cutouts': random_center_cutouts,
        }

        return sample


class CropDataset_pred(Dataset):
    def __init__(self, csv_file, root_dir='.'):
        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, 'all_images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        coor_label = np.array(self.all_data.iloc[idx, 1:3]).astype(int)
        coor_pred = np.array(self.all_data.iloc[idx, 3:]).astype(int)
        coor_crop = np.copy(image_np)
        coor_crop = crop_image(coor_crop, coor_pred)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        coor_crop = input_transform(coor_crop)
        # image = input_transform(image_pil)

        sample = {
            'image_name': image_name,
            'coor_crop': coor_crop,
            'coor_label': coor_label,
            'coor_pred': coor_pred,
        }

        return sample
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # dataset = InvalidDataset('csv/pass_invalid_85.csv')
    # dataset = CutoutDataset('csv/pass_valid_test_100.csv')
    dataset = CutoutDataset('csv/pass_valid_50.csv')
    # dataset = CropDataset_pred('csv/pred_pass_valid_test_100.csv')

    dataloader = DataLoader(dataset, batch_size=4)
    iter_data = iter(dataloader)
    batch = next(iter_data)
    coor_cutout = batch['coor_cutout']
    # image_name = batch['image_name']
    # coor_label = batch['coor_label']
    # coor_pred = batch['coor_pred']
    # coor_crop = batch['coor_crop']

    # ===============testing======================
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('tkagg')
    plt.imshow(coor_cutout[0].numpy().reshape(224, 224))
    plt.show()
    # ===============testing======================
