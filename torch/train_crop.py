from Classifier_cutout import Classifier
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import CropDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
from random import uniform
from utils import heatmap_to_coor, accuracy_sum, spike, gaussion_hmap
from torch.utils.data.dataset import Subset
from tqdm import tqdm

# from test_cutout_cls import eval_cutout_cls

from pdb import set_trace


training_number = 200
num_epochs = 1000000
batch_size = 2
invalid_batch_size = batch_size
train_csv = './csv/pass_valid_200.csv'
# invalid_csv = './csv/pass_invalid_85.csv'

# writer_dir = 'runs/classification_crop_' + str(training_number)
dir_weight = 'check_points/classifier_crop_50.pth'
classifier = Classifier().cuda()

dist_lower_bound = 10.0
train_dataset = CropDataset(csv_file=train_csv)

#subset of the dataset
indices = list(range(len(train_dataset)//4))
subset = Subset(train_dataset, indices)
train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# invalid_dataset = InvalidDataset(csv_file=invalid_csv)
# invalid_loader = DataLoader(invalid_dataset, batch_size=invalid_batch_size, shuffle=True)

lr = 1e-5

# writer = SummaryWriter(writer_dir)


# criterion = nn.MSELoss().cuda()
criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.functional.binary_cross_entropy()


# G_solver = torch.optim.Adam(student.parameters(), lr=lr_G)
solver = torch.optim.Adam(classifier.parameters(), lr=lr)
# G_solver = torch.optim.RMSprop(student.parameters(), lr=lr_G)
# solver = torch.optim.RMSprop(classifier.parameters(), lr=lr_D)
# G_solver = torch.optim.Adadelta(student.parameters(), lr=lr_G)
# solver = torch.optim.Adadelta(classifier.parameters(), lr=lr_D)

def train():
    correct_rate = 0
    max_e_distance = 9999
    # invalid_iter = iter(invalid_loader)
    # invalid_batch = next(invalid_iter)
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        # pbar = tqdm(total=len(train_loader))
        for i, batch_data in enumerate(train_loader):
            classifier.zero_grad()

            # try:
                # image_invalid = invalid_batch['image'].cuda()
                # image_invalid_cutouts = invalid_batch['random_crop'].cuda().permute(1,0,2,3,4)

            # except:
                # invalid_iter = iter(invalid_loader)
                # invalid_batch = next(invalid_iter)
                # image_invalid = invalid_batch['image'].cuda()
                # image_invalid_cutouts = invalid_batch['random_crop'].cuda().permute(1,0,2,3,4)

            # image_valid = batch_data['image'].cuda()
            image_valid_coor_crop = batch_data['coor_crop'].cuda()
            image_valid_rand_crop = batch_data['random_crop'].cuda().permute(1,0,2,3,4)

            # b_valid_size = len(image_valid)
            b_valid_coor_crop_size = len(image_valid_coor_crop)
            b_valid_rand_crop_size = image_valid_rand_crop.shape[0] * image_valid_rand_crop.shape[1]
            # b_invalid_size = len(image_invalid)
            # b_invalid_cut_size = image_invalid_cutouts.shape[0] * image_invalid_cutouts.shape[1]
            
            for idx, image in enumerate(image_valid_rand_crop):
                logits_valid_rand_crop = classifier(image)
                label_valid_rand_crop_zeros = torch.zeros(len(image), 1).cuda()
                correct += (logits_valid_rand_crop<0.5).sum().item() 
                if idx == 0:
                    loss_valid_rand_crop = criterion(logits_valid_rand_crop, label_valid_rand_crop_zeros)
                else:
                    loss_valid_rand_crop = loss_valid_rand_crop \
                                        + criterion(logits_valid_rand_crop, label_valid_rand_crop_zeros)

            # for idx, image in enumerate(image_invalid_cutouts):
                # logits_invalid_cut = classifier(image)
                # label_invalid_cut_zeros = torch.zeros(len(image), 1).cuda()
                # correct += (logits_invalid_cut<0.5).sum().item() 
                # if idx == 0:
                    # loss_invalid_cut = criterion(logits_invalid_cut, label_invalid_cut_zeros)
                # else:
                    # loss_invalid_cut = loss_invalid_cut \
                                    # + criterion(logits_invalid_cut, label_invalid_cut_zeros)

            # logits_valid = classifier(image_valid)
            logits_valid_coor_crop = classifier(image_valid_coor_crop)
            # logits_invalid = classifier(image_invalid)

            # label_valid_ones = torch.ones(b_valid_size, 1).cuda()
            label_valid_coor_crop_ones= torch.ones(b_valid_coor_crop_size, 1).cuda()
            # label_invalid_zeros = torch.zeros(b_invalid_size, 1).cuda()

            # loss_valid = criterion(logits_valid, label_valid_ones)
            loss_valid_coor_crop = criterion(logits_valid_coor_crop, label_valid_coor_crop_ones)
            # loss_invalid = criterion(logits_invalid, label_invalid_zeros)

            loss =  loss_valid_coor_crop \
                    + loss_valid_rand_crop  \
                    # + loss_valid \
                    # + loss_invalid \
                    # + loss_invalid_cut \

            loss.backward()
            solver.step()
        
            correct = correct  \
                    + (logits_valid_coor_crop>=0.5).sum().item() \
                    # + (logits_valid_rand_crop<0.5).sum().item() \
                    # + (logits_valid>=0.5).sum().item() \
                    # + (logits_invalid<0.5).sum().item() 
                    # + (logits_invalid_cut<0.5).sum().item() 

            total = total \
                    + b_valid_coor_crop_size  \
                    + b_valid_rand_crop_size  \
                    # + b_valid_size \
                    # + b_invalid_size  \
                    # + b_invalid_cut_size \

            # pbar.update()

    # if (epoch+1) % 1 == 0:    # every 20 mini-batches...
        print('Train epoch {}:\tD_loss: {:.10f} acc_training_D: {:.3f}%  {}/{}'.format(
                epoch+1,
                # G_loss.item(),
                loss.item(),
                100 * correct / total,
                correct,
                total))


        torch.save(classifier.state_dict(), dir_weight)
        print('model saved to ' + dir_weight)
        # writer.add_scalar("advererial_gan_G_loss", \
                # G_loss.item(), #/ len(image), \
                # epoch * math.ceil(len(welding_train_loader) / batch_size) \
                # )

        # if (epoch+1) % 5 == 0:    # every 20 mini-batches...
            # eval_cutout_cls()

if __name__ == "__main__":
    train()

