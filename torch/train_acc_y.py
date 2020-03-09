from Teacher import Teacher
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import WeldingDatasetToTensor, FakeData 
from pdb import set_trace
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
from random import uniform
from utils import heatmap_to_coor, accuracy_sum, spike

training_number = 1000
train_csv = './csv/generated.csv'

# weight_to_load_student = './check_points/weights_300.pth'


# weight_to_load_teacher = './check_points/weights_adverserial_teacher_39.pth'
# writer_student_dir = 'runs/adverserial_student_' + str(training_number)


writer_gan_dir = 'runs/train_acc_' + str(training_number)


teacher = Teacher().cuda()
# teacher.load_state_dict(torch.load(weight_to_load_teacher))
teacher.train()


train_dataset = FakeData(csv_file=train_csv, root_dir='./')
# valid_dataset = WeldingDatasetToTensor(csv_file=val_csv, root_dir='./')

num_epochs = 1000000
batch_size = 10
# batch_size_valid = 4
lr_D = 1e-7
# lr_G = 1e-4

# welding_valid_loader = DataLoader(valid_dataset, batch_size=batch_size, \
                        # num_workers=4, shuffle=True)

# writer_teacher = SummaryWriter(writer_teacher_dir)
# writer_student = SummaryWriter(writer_student_dir)
writer_gan = SummaryWriter(writer_gan_dir)


# criterion = nn.MSELoss().cuda()
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()

D_solver = torch.optim.Adam(teacher.parameters(), lr=lr_D)
# D_solver = torch.optim.RMSprop(teacher.parameters(), lr=lr_D)
# D_solver = torch.optim.Adadelta(teacher.parameters(), lr=lr_D)

def train_gan():
    loss = 999


    for epoch in range(num_epochs):
        welding_train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                                num_workers=4, shuffle=True)

        # print('Start epoch {} training on teacher'.format(epoch))
        # for _ in range(5):
        dataloader_iterator = iter(welding_train_loader)
        try:
            sample_batched= next(dataloader_iterator)
        except StopIteration:
            welding_train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                                num_workers=4, shuffle=True)
            dataloader_iterator = iter(welding_train_loader)
            sample_batched = next(dataloader_iterator)
        # for i_batch, sample_batched in enumerate(welding_train_loader):
        # if (i_batch + 1) > math.ceil(training_number / batch_size):
            # break
        inputs = sample_batched['image']#.cuda()
        real_hmaps = sample_batched['hmap']#.cuda()
        fake_hmaps = sample_batched['hmap_fake']#.cuda()
        coors_real = sample_batched['coor_bc']#.cuda()
        coors_fake = sample_batched['coor_bc_fake']#.cuda()

        # pred_bc = sample_batched['pred_bc']#.cuda()
        # acc_xy = sample_batched['acc_xy']#.cuda()

        empty_batch = True
        # all_acc_x = []
        # all_acc_y = []

        # label = torch.zeros(batch_size, 2).cuda()
        label = np.zeros((batch_size, 1))


        for idx, input in enumerate(inputs):
            input = input.cuda()

            # x, y = heatmap_to_coor(pseudo_hmap.cpu().detach().numpy().reshape(224, 224))
            
            # acc_x = abs(int(coors_fake[idx][0])-coors_real[idx][0].item()) / 1280
            # label[idx, 0] = acc_x
            acc_y = abs(int(coors_fake[idx][1])-coors_real[idx][1].item()) / 1024
            label[idx, 0] = acc_y



            # all_acc_x.append(acc_x)
            # all_acc_y.append(acc_y)
            # label[idx, 0] = acc_y
            


            # pseudo_hmap = spike(fake_hmaps[idx]).cuda()
            pseudo_hmap = fake_hmaps[idx].cuda()
            # real_hmap = spike(real_hmaps[idx]).cuda()

            pseudo = torch.cat([input, pseudo_hmap.float()], dim=0).unsqueeze(0)
            # real = torch.cat([input.float(), real_hmap.float()], dim=0).unsqueeze(0)


            if empty_batch:
                pseudo_batch = pseudo
                # real_batch = real
                empty_batch = False
            else:
                pseudo_batch = torch.cat([pseudo_batch, pseudo], dim=0)
                # real_batch = torch.cat([real_batch, real]) 

        # ones_label = torch.ones(batch_size, 1).cuda()
        label = torch.from_numpy(label).float().cuda()

        # D_real = teacher(real_batch)
        D_fake = teacher(pseudo_batch)
        
        # D_loss_fake = nn.functional.binary_cross_entropy(D_fake, label)
        # D_loss = nn.functional.mse_loss(D_fake, label)
        # D_loss = nn.functional.binary_cross_entropy(D_fake, label)
        D_loss = nn.functional.mse_loss(D_fake, label)
        # D_loss = D_loss_real + D_loss_fake# + alpha * student_loss

        # D_loss.backward(retain_graph=True)
        D_loss.backward()
        D_solver.step()
        # G_solver.step()

        # G_solver.zero_grad()
        D_solver.zero_grad()
        # correct += (D_real>=0.5).sum().item() + (D_fake<0.5).sum().item()
        # total += batch_size * 2



        if (epoch+1) % 20 == 0:    # every 20 mini-batches...

            print('Train epoch {}:\tD_loss: {:.30f} '.format(
                    epoch+1,
                    # G_loss.item(),
                    D_loss.item()))
                    # 100 * correct / total,
                    # correct,
                    # total)


            # writer_gan.add_scalar("advererial_gan_G_loss", \
                    # G_loss.item(), #/ len(inputs), \
                    # epoch * math.ceil(len(welding_train_loader) / batch_size) \
                    # )

            writer_gan.add_scalar("train_acc generated_random_60K", \
                    D_loss.item(), #/ len(inputs), \
                    epoch * math.ceil(len(welding_train_loader) / batch_size) \
                    )

        if (epoch+1) % 100 == 0:    # every 20 mini-batches...

            if D_loss.item() < loss:
                loss = D_loss.item()
                
                dir_weight = './check_points/weights_train_acc_y.pth'
                # dir_weight = './check_points/weights_train_acc_y.pth'
                torch.save(teacher.state_dict(), dir_weight)
                print('model saved to ' + dir_weight)


train_gan()

