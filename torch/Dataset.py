import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
from skimage import io, transform
# from utils import *
# from torchvision import transforms, utils
from torch.utils.data import DataLoader
from pdb import set_trace
from torch.utils.data.sampler import SubsetRandomSampler
from utils import CoorToHeatmap
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image

# Helper function to show a batch
def show_coor_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, coor_batch = \
            sample_batched['image'], sample_batched['coor']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    # set_trace()

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(coor_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    coor_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

class WeldingDataset(Dataset):
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
        self.input_transform = Compose([Resize((224, 224)), ToTensor()])
        # self.target_transform = Compose([CoorToHeatmap(224), ToTensor()])
        self.to_heatmap_transform = CoorToHeatmap(224)
        self.target_to_tensor_transform = ToTensor()

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir + 'images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image = np.array(image_pil)
        # image = io.imread(img_name)
        # image = image.astype(np.float32)
        # image = Image.fromarray(np.uint8(image))
        coor = self.all_data.iloc[idx, 1:]
        coor = np.array([coor])
        # coor = coor.astype(np.float32).reshape(-1, 2)
        sample = {'image': image, 'coor': coor}
        sample = self.to_heatmap_transform(sample)
        coor = sample['coor']
        coor = self.target_to_tensor_transform(coor)
        image = self.input_transform(image_pil)
        # coor = self.target_to_tensor_transform(coor)
        sample = {'image': image, 'coor': coor}
        # set_trace()

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
    rescale = Rescale(224)
    to_tensor = ToTensor()
    padding = Padding()
    coor_to_heatmap = CoorToHeatmap()
    composed = transforms.Compose([rescale, padding, coor_to_heatmap, to_tensor])


    transformed_dataset = WeldingDataset(csv_file='./saved_dict.csv', root_dir='./', \
                                        transform=composed)

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].size(), sample['coor'].size())

        if i == 3:
            break

###########################################    Dataloader 

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, \
                            num_workers=4)

    random_seed = 42
    split = 1000
    shuffle_dataset = True
    indices = list(range(len(transformed_dataset)))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[:-split], indices[-split:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(transformed_dataset, batch_size=4, \
                            num_workers=4, sampler=train_sampler)
    valid_loader = DataLoader(transformed_dataset, batch_size=4, \
                            num_workers=8, sampler=valid_sampler)


    for i_batch, sample_batched in enumerate(valid_loader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['coor'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_coor_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break




