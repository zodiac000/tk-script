from Student_bc import Student
from Teacher import Teacher
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import WeldingDatasetToTensor 
from pdb import set_trace
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
from random import uniform
from utils import heatmap_to_coor, accuracy_sum, spike, gaussion_hmap

training_number = 1000
train_csv = './csv/pass_valid_tail_1000.csv'
# val_csv = './csv/pass_valid_1000_validation.csv'

weight_to_load_student = './check_points/weights_1000.pth'
# writer_teacher_dir = 'runs/adverserial_teacher_' + str(training_number)


# weight_to_load_teacher = './check_points/weights_adverserial_teacher_39.pth'
# writer_student_dir = 'runs/adverserial_student_' + str(training_number)


writer_gan_dir = 'runs/adverserial_gan_' + str(training_number)

student = Student().cuda()
student.load_state_dict(torch.load(weight_to_load_student))

teacher = Teacher().cuda()
# teacher.load_state_dict(torch.load(weight_to_load_teacher))
# student.train()
teacher.train()


dist_lower_bound = 10.0
train_dataset = WeldingDatasetToTensor(csv_file=train_csv, root_dir='./', dist_lower_bound=dist_lower_bound)
# valid_dataset = WeldingDatasetToTensor(csv_file=val_csv, root_dir='./')

num_epochs = 1000000
batch_size = 4
# batch_size_valid = 4
lr_D = 1e-6
lr_G = 1e-4

writer_gan = SummaryWriter(writer_gan_dir)


# criterion = nn.MSELoss().cuda()
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()

# G_solver = torch.optim.Adam(student.parameters(), lr=lr_G)
D_solver = torch.optim.Adam(teacher.parameters(), lr=lr_D)
# G_solver = torch.optim.RMSprop(student.parameters(), lr=lr_G)
# D_solver = torch.optim.RMSprop(teacher.parameters(), lr=lr_D)
# G_solver = torch.optim.Adadelta(student.parameters(), lr=lr_G)
# D_solver = torch.optim.Adadelta(teacher.parameters(), lr=lr_D)

def train_gan():
    correct_rate = 0
    max_e_distance = 9999
    correct = 0
    total = 0
    for epoch in range(num_epochs):
        welding_train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                                num_workers=0, shuffle=True)

        dataloader_iterator = iter(welding_train_loader)
        try:
            sample_batched= next(dataloader_iterator)
        except StopIteration:
            welding_train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                                num_workers=0, shuffle=True)
            dataloader_iterator = iter(welding_train_loader)
            sample_batched = next(dataloader_iterator)
        inputs = sample_batched['image']#.cuda()
        real_hmaps = sample_batched['hmap']#.cuda()
        coors_bc = sample_batched['coor_bc']
        random_hmap1 = sample_batched['random_hmap1']
        # random_coor1 = sample_batched['random_coor1']
        random_hmap2 = sample_batched['random_hmap2']
        # random_coor2 = sample_batched['random_coor2']
        random_hmap3 = sample_batched['random_hmap3']
        # random_coor3 = sample_batched['random_coor3']
        # random_hmap4 = sample_batched['random_hmap4']
        # random_coor4 = sample_batched['random_coor4']
        # inames = sample_batched['img_name']


        pseudo_batch = None
        real_batch = None
        for idx, input in enumerate(inputs):
            input = input.cuda()

            pseudo_hmap = student(input.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            # pseudo_hmap = spike(pseudo_hmap).cuda()
            # real_hmap = spike(real_hmaps[idx]).cuda()
            # pseudo_hmap = spike(pseudo_hmap) 
            _, y, x = np.unravel_index(pseudo_hmap.argmax(), pseudo_hmap.shape)
            pseudo_hmap = gaussion_hmap(x, y)
            pseudo_hmap = torch.from_numpy(pseudo_hmap)
            pseudo_hmap = nn.Softmax(1)(pseudo_hmap.view(-1, 224*224)).view(1, 224, 224)
            pseudo_hmap = pseudo_hmap.cuda()
            real_hmap = real_hmaps[idx].cuda()

            pseudo = torch.cat([input, pseudo_hmap.float()], dim=0).unsqueeze(0)
            real = torch.cat([input.float(), real_hmap.float()], dim=0).unsqueeze(0)

            x, y = heatmap_to_coor(pseudo_hmap.detach().cpu().numpy().reshape(224, 224))
            e_distance = ((int(x/224*1280) - coors_bc[idx][0].item())**2 + \
                         (int(y/224*1024)-coors_bc[idx][1].item())**2)**0.5   
            if e_distance < dist_lower_bound:
                real = torch.cat([real, pseudo]) 
                # pseudo_first = torch.cat([input, spike(random_hmap1[idx]).cuda()], dim=0).unsqueeze(0)
                # pseudo_second = torch.cat([input, spike(random_hmap2[idx]).cuda()], dim=0).unsqueeze(0)
                pseudo_first = torch.cat([input, random_hmap1[idx].float().cuda()], dim=0).unsqueeze(0)
                pseudo_second = torch.cat([input, random_hmap2[idx].float().cuda()], dim=0).unsqueeze(0)
                pseudo_third = torch.cat([input, random_hmap3[idx].float().cuda()], dim=0).unsqueeze(0)
                # pseudo_forth = torch.cat([input, random_hmap4[idx].float().cuda()], dim=0).unsqueeze(0)
                # pseudo = torch.cat([pseudo_first, pseudo_second, pseudo_third, pseudo_forth]) 
                pseudo = torch.cat([pseudo_first, pseudo_second, pseudo_third]) 

            try:
                pseudo_batch = torch.cat([pseudo_batch, pseudo], dim=0)
                real_batch = torch.cat([real_batch, real]) 
            except:
                pseudo_batch = pseudo
                real_batch = real

        D_real = teacher(real_batch)
        D_fake = teacher(pseudo_batch)
        ones_label = torch.ones(real_batch.shape[0], 1).cuda()
        zeros_label = torch.zeros(pseudo_batch.shape[0], 1).cuda()

        D_loss_real = nn.functional.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = nn.functional.binary_cross_entropy(D_fake, zeros_label)

        # D_loss_real = nn.functional.mse_loss(D_real, ones_label)
        # D_loss_fake = nn.functional.mse_loss(D_fake, zeros_label)

        D_loss = D_loss_real + D_loss_fake


        D_loss.backward()
        D_solver.step()

        D_solver.zero_grad()
    
        # correct += (torch.max(predicted, 1)[1] == torch.max(labels, 1)[1]).sum().item()
        correct += (D_real>=0.5).sum().item() + (D_fake<0.5).sum().item()
        # set_trace()
        total += real_batch.shape[0] + pseudo_batch.shape[0]




        if (epoch+1) % 20 == 0:    # every 20 mini-batches...

            print('Train epoch {}:\tD_loss: {:.30f} acc_training_D: {:.30f}%  {}/{}'.format(
                    epoch,
                    # G_loss.item(),
                    D_loss.item(),
                    100 * correct / total,
                    correct,
                    total))


            # writer_gan.add_scalar("advererial_gan_G_loss", \
                    # G_loss.item(), #/ len(inputs), \
                    # epoch * math.ceil(len(welding_train_loader) / batch_size) \
                    # )

            writer_gan.add_scalar("advererial_gan_D_loss", \
                    D_loss.item(), #/ len(inputs), \
                    epoch * math.ceil(len(welding_train_loader) / batch_size) \
                    )

            writer_gan.add_scalar("adverserial_training_D_acc", \
                    100 * correct / total, #/ len(inputs), \
                    epoch * math.ceil(len(welding_train_loader) / batch_size) \
                    )
            correct = 0 
            total = 0
        if (epoch+1) % 100 == 0:    # every 20 mini-batches...
            dir_weight = './check_points/weights_adv.pth'
            torch.save(teacher.state_dict(), dir_weight)
            print('model saved to ' + dir_weight)

train_gan()

