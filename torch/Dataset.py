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
    print(grid_image.shape)
    grid = np.concatenate((grid_image.numpy(), grid_coor.numpy()), axis=1)
    print(grid.shape)
    plt.imshow(grid.transpose((1, 2, 0)))


class WeldingDatasetToTensor(Dataset):
    """Welding dataset."""

    def __init__(self, csv_file, root_dir, dist_lower_bound):
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

        hmap = CoorToHeatmap(image_np.shape, 224)(coor_bc)
        hmap = torch.from_numpy(hmap)
        hmap = nn.Softmax(2)(hmap.view(-1, 1, 224*224)).view(1, 224, 224)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        
        image = input_transform(image_pil)
        
        #generating random fake heat map
        while(True):
            random_coor1 = np.asarray([randint(200, 800), randint(200, 800)])
            dist = ((coor_bc[0] - random_coor1[0])**2 + (coor_bc[1] - random_coor1[1])**2)**0.5
            if dist > self.dist_lower_bound:
                break
        random_hmap1 = CoorToHeatmap(image_np.shape, 224)(random_coor1)
        random_hmap1 = torch.from_numpy(random_hmap1)
        random_hmap1 = nn.Softmax(2)(random_hmap1.view(-1, 1, 224*224)).view(1, 224, 224)
    
        while(True):
            random_coor2 = np.asarray([randint(200, 800), randint(200, 800)])
            dist = ((coor_bc[0] - random_coor2[0])**2 + (coor_bc[1] - random_coor2[1])**2)**0.5
            if dist > self.dist_lower_bound:
                break

        random_hmap2 = CoorToHeatmap(image_np.shape, 224)(random_coor2)
        random_hmap2 = torch.from_numpy(random_hmap2)
        random_hmap2 = nn.Softmax(2)(random_hmap2.view(-1, 1, 224*224)).view(1, 224, 224)

        while(True):
            random_coor3 = np.asarray([randint(200, 800), randint(200, 800)])
            dist = ((coor_bc[0] - random_coor3[0])**2 + (coor_bc[1] - random_coor3[1])**2)**0.5
            if dist > self.dist_lower_bound:
                break

        random_hmap3 = CoorToHeatmap(image_np.shape, 224)(random_coor3)
        random_hmap3 = torch.from_numpy(random_hmap3)
        random_hmap3 = nn.Softmax(2)(random_hmap3.view(-1, 1, 224*224)).view(1, 224, 224)

        while(True):
            random_coor4 = np.asarray([randint(200, 800), randint(200, 800)])
            dist = ((coor_bc[0] - random_coor4[0])**2 + (coor_bc[1] - random_coor4[1])**2)**0.5
            if dist > self.dist_lower_bound:
                break

        random_hmap4 = CoorToHeatmap(image_np.shape, 224)(random_coor4)
        random_hmap4 = torch.from_numpy(random_hmap4)
        random_hmap4 = nn.Softmax(2)(random_hmap4.view(-1, 1, 224*224)).view(1, 224, 224)

        sample = {
                'image': image, \
                'hmap': hmap, \
                'img_name': image_name, \
                'coor_bc': coor_bc, \
                'origin_img': image_np,\
                'random_hmap1': random_hmap1, \
                'random_coor1': random_coor1, \
                'random_hmap2': random_hmap2, \
                'random_coor2': random_coor2, \
                'random_hmap3': random_hmap3, \
                'random_coor3': random_coor3, \
                'random_hmap4': random_hmap4, \
                'random_coor4': random_coor4, \
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
        hmap_fake = nn.Softmax(2)(hmap_fake.view(-1, 1, 224*224)).view(1, 224, 224)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        
        # set_trace()
        image = input_transform(image_pil)
    
        sample = {'image': image, \
                'hmap': hmap, \
                'img_name': image_name, \
                'coor_bc': coor_bc, \
                'origin_img': image_np,\
                'coor_bc_fake': coor_bc_fake,\
                'hmap_fake': hmap_fake, \

                # 'pred_bc': pred_bc,\
                # 'acc_xy': acc_xy,\

                # 'class_real': class_real, \
                }
                # 'class_real': class_real, 'dx_dy':dx_dy}
        
        return sample


if __name__ == "__main__":
###########################################   Not Transformed Dataset
    # welding_dataset = WeldingDataset(csv_file='./saved_dict.csv', root_dir='./')
    # for i in range(len(welding_dataset)):
        # sample = welding_dataset[i]

        # print(i, sample['image'].shape, sample['coor'])
        
        # rows = 3
        # colums = 4
        # ax = plt.subplot(rows, colums, i + 1)
        # plt.tight_layout()
        # ax.set_title('Sample #{}'.format(i))
        # ax.axis('off')
        # show_coordinate(**sample)


        # if i == rows * colums - 1:
            # plt.show()
            # break


###########################################    Transformed Dataset


    transformed_dataset = WeldingDataset(csv_file='./csv/saved_dict.csv', root_dir='./')

    # for i in range(len(transformed_dataset)):
        # sample = transformed_dataset[i]

        # print(i, sample['image'].size(), sample['coor'].size())

        # if i == 3:
            # break

###########################################    Dataloader 

    # dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, \
                            # num_workers=4)

    # random_seed = 42
    # split = 1000
    # shuffle_dataset = True
    # indices = list(range(len(transformed_dataset)))
    # train_indices, val_indices = indices[:-split], indices[-split:]

    # if shuffle_dataset:
        # np.random.seed(random_seed)
        # np.random.shuffle(train_indices)
    # train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(transformed_dataset, batch_size=4, \
                            num_workers=4)
    # valid_loader = DataLoader(transformed_dataset, batch_size=4, \
                            # num_workers=4)


