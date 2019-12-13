from Student_reg import Student
# from Student2 import Student2
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import WeldingDatasetToTensor
from pdb import set_trace
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from utils import accuracy_sum, heatmap_to_coor, crop
from torchvision.transforms import ToTensor, Resize
from PIL import Image

# training_number = 50
# training_number = 4265
training_number = 6500
# training_number = 6415

train_dataset = WeldingDatasetToTensor(csv_file='./csv/pass_adbc_' + str(training_number) + '.csv', \
                                        root_dir='./')
# val_dataset = WeldingDatasetToTensor(csv_file='./csv/pass_valid_1000.csv', root_dir='./')
val_dataset = WeldingDatasetToTensor(csv_file='./csv/pass_adbc_1000.csv', root_dir='./')
# val_dataset = WeldingDatasetToTensor(csv_file='./csv/tail_100.csv', root_dir='./')
saved_weight_dir = './check_points/saved_weights_reg_adbc_' + str(training_number) + '.pth'
tensorboard_file = 'runs/cas_reg_adbc_' + str(training_number)

# saved_weights = './check_points/saved_weights_200.pth'
for i in range(len(train_dataset)):
    sample = train_dataset[i]
    print(i, sample['image'].size(), sample['hmap'].size())
    if i == 3:
        break

num_epochs = 30000
batch_size = 20
lr = 1e-5

train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                        num_workers=4, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, \
                        num_workers=4)

writer = SummaryWriter(tensorboard_file)


model = Student().cuda()
# model = Student2().cuda()
mse = nn.MSELoss().cuda()
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
l = 0.05
def train():
    train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                        num_workers=4, shuffle=True)
    # model.load_state_dict(torch.load(saved_weights))
    max_total_acc_x = 0
    max_euclidean_distance = 99999
    for epoch in range(num_epochs):
        dataloader_iterator = iter(train_loader)
        try:
            sample_batched= next(dataloader_iterator)
        except StopIteration:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                                num_workers=4, shuffle=True)
            dataloader_iterator = iter(train_loader)
            sample_batched = next(dataloader_iterator)

        inputs = sample_batched['image'].cuda()
        labels = sample_batched['hmap'].cuda()
        coors_bc = sample_batched['coor_bc'].cpu().detach().numpy()
        class_real = sample_batched['class_real'].cuda()
        gt_dx_dy = sample_batched['dx_dy'].numpy()

        img_names = sample_batched['img_name']
        origin_imgs = sample_batched['origin_img']

        w = origin_imgs.shape[2]
        h = origin_imgs.shape[1]

        dx = gt_dx_dy[:, :1] / w * 224
        dy = gt_dx_dy[:, 1:] / h * 224

        gt_dx_dy = torch.from_numpy(np.concatenate((dx, dy), axis=1)).cuda().float()

        optimizer.zero_grad()
        outputs, pred_dx_dy = model(inputs)

        # loss_hmap = mse(outputs, labels)
        loss = mse(pred_dx_dy, gt_dx_dy)
        # loss = loss_hmap + loss_dx_dy

        # loss_bce = nn.functional.binary_cross_entropy(class_pred, class_real)
        # loss = (1-class_real) * loss_bce + class_real * (l * loss_bce + (1-l)*loss_mse)
        torch.mean(loss).backward()
        optimizer.step()
        

        if (epoch+1) % 5 == 0:    # every 20 mini-batches...
            print('Train epoch: {}\tLoss: {:.30f}'.format(
                    epoch+1,
                    torch.mean(loss).item())) #/ len(inputs)))

        if (epoch+1) % 50 == 0:    # every 20 mini-batches...
            # model.eval()
            with torch.no_grad():
                valid_loss = 0
                total_acc_x = 0
                total_acc_y = 0
                e_distance = 0
                for i, batch in enumerate(valid_loader):
                    inputs = batch['image'].float().cuda()
                    labels = batch['hmap'].float().cuda()

                    coors_bc = sample_batched['coor_bc'].cpu().detach().numpy()
                    gt_dx_dy = sample_batched['dx_dy'].numpy()

                    w = origin_imgs.shape[2]
                    h = origin_imgs.shape[1]

                    dx = gt_dx_dy[:, :1] / w * 224
                    dy = gt_dx_dy[:, 1:] / h * 224

                    gt_dx_dy = torch.from_numpy(np.concatenate((dx, dy), axis=1)).cuda().float()
                    
                    
                    outputs, pred_dx_dy = model(inputs)
                    # loss_hmap = mse(outputs, labels)
                    loss = mse(pred_dx_dy, gt_dx_dy)
                    # valid_loss += loss_hmap + loss_dx_dy
                    # loss_mse = mse(outputs, labels.cuda())
                    # loss_bce = nn.functional.binary_cross_entropy(class_pred, class_real)
                    # loss = (1-class_real) * loss_bce + class_real * (l * loss_bce + (1-l)*loss_mse)
                    # valid_loss += torch.mean(loss)


                valid_loss = valid_loss / len(valid_loader)
                print('Valid loss {}'.format(valid_loss))


train()
