import torch
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
from pdb import set_trace

# Helper function to show a batch
def show_coor_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, coor_batch = \
            sample_batched['image'], sample_batched['coor']
    # batch_size = len(images_batch)
    # im_size = images_batch.size(2)
    # grid_border_size = 2

    grid_image = utils.make_grid(images_batch)
    grid_coor = utils.make_grid(coor_batch)
    print(grid_image.shape)
    grid = np.concatenate((grid_image.numpy(), grid_coor.numpy()), axis=1)
    print(grid.shape)
    plt.imshow(grid.transpose((1, 2, 0)))

    # for i in range(batch_size):
        # plt.scatter(coor_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    # coor_batch[i, :, 1].numpy() + grid_border_size,
                    # s=10, marker='.', c='r')

        # plt.title('Batch from dataloader')

class WeldingDatasetToTensor(Dataset):
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
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]        
        img_name = os.path.join(self.root_dir + 'images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        origin_image = np.array(image_pil)
        coor = self.all_data.iloc[idx, 1:]
        coor = np.array(coor)
        sample = {'image': origin_image, 'coor': coor}
        hmap = CoorToHeatmap(224)(sample)
        hmap = ToTensor()(hmap)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        image = input_transform(image_pil)
    
        sample = {'image': image, 'hmap': hmap, \
                'img_name': image_name, 'coor': coor.astype(int), 'origin_img': origin_image}
        
        return sample

# class WeldingDataset(Dataset):
    # """Welding dataset."""

    # def __init__(self, csv_file, root_dir):
        # """
        # Args:
            # csv_file (string): Path to the csv file with annotations.
            # root_dir (string): Directory with all the images.
            # transform (callable, optional): Optional transform to be applied
                # on a sample.
        # """

        # self.all_data = pd.read_csv(csv_file, header=None)
        # self.root_dir = root_dir
        # self.CoorToHeatmap = CoorToHeatmap(224)
        # self.Resize = Resize((224, 224))
        # self.input_transform = Compose([Resize((224, 224)), ToTensor()])

    # def __len__(self):
        # return len(self.all_data)


    # def __getitem__(self, idx):
        # if torch.is_tensor(idx):
            # idx = idx.tolist()
        # image_name = self.all_data.iloc[idx, 0]        
        # img_name = os.path.join(self.root_dir + 'images/',
                                # self.all_data.iloc[idx, 0])
        # image_pil = Image.open(img_name)
        # image = np.array(image_pil)
        # coor = self.all_data.iloc[idx, 1:]
        # coor = np.array([coor])
        # sample = {'image': image, 'coor': coor}
        # sample = self.CoorToHeatmap(sample)
        # coor = sample['coor']
        # coor = self.Resize(coor)
        # coor = np.array(coor)
        # image = self.input_transform(image_pil)
        # sample = {'image': image, 'coor': coor, 'img_name': image_name}
        # return sample



# class MixedDataset(WeldingDataset):
    # def __init__(self, csv_file, root_dir, student):
        # super().__init__(csv_file, root_dir)
        # self.student = student
    # def __getitem__(self, idx):
        # sample = self.helper(idx)
        
        # if np.random.uniform() < 0.5: # dice 5 - 5 
            # # Real label
            # return {'image': cat([sample['image'], sample['coor']], dim=1), 'label': torch.ones(1)}  # N, 2, H, W
        # else:
            # # Pseudo label
            # return {'image': cat([sample['image'].cuda(), self.student(sample['image'].unsqueeze(0).cuda()).squeeze(0)], dim=1), 
                    # 'label': torch.zeros(1)} # return image with student-generated label




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


