from Student import Student
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

training_number = 200
# train_csv = './csv/head_4265.csv'
train_csv = './csv/head_' + str(training_number) + '.csv'
val_csv = './csv/tail_1000.csv'

weight_to_load_student = './check_points/saved_weights_' + str(training_number) + '.pth'
# writer_teacher_dir = 'runs/adverserial_teacher_' + str(training_number)


# weight_to_load_teacher = './check_points/weights_adverserial_teacher_39.pth'
# writer_student_dir = 'runs/adverserial_student_' + str(training_number)


writer_gan_dir = 'runs/adverserial_gan_' + str(training_number)

student = Student().cuda()
student.load_state_dict(torch.load(weight_to_load_student))

teacher = Teacher().cuda()
# teacher.load_state_dict(torch.load(weight_to_load_teacher))

train_dataset = WeldingDatasetToTensor(csv_file=train_csv, root_dir='./')
valid_dataset = WeldingDatasetToTensor(csv_file=val_csv, root_dir='./')

num_epochs = 200
batch_size = 4
# batch_size_valid = 4
lr_D = 1e-4
lr_G = 1e-4

welding_valid_loader = DataLoader(valid_dataset, batch_size=batch_size, \
                        num_workers=4, shuffle=True)

# writer_teacher = SummaryWriter(writer_teacher_dir)
# writer_student = SummaryWriter(writer_student_dir)
writer_gan = SummaryWriter(writer_gan_dir)


# criterion = nn.MSELoss().cuda()
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()

# G_solver = torch.optim.Adam(student.parameters(), lr=lr_G)
# D_solver = torch.optim.Adam(teacher.parameters(), lr=lr_D)
G_solver = torch.optim.RMSprop(student.parameters(), lr=lr_G)
D_solver = torch.optim.RMSprop(teacher.parameters(), lr=lr_D)
# G_solver = torch.optim.Adadelta(student.parameters(), lr=lr_G)
# D_solver = torch.optim.Adadelta(teacher.parameters(), lr=lr_D)

def train_gan():
    correct_rate = 0
    for epoch in range(num_epochs):
        welding_train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                                num_workers=4, shuffle=True)
        correct = 0
        total = 0

        print('Start epoch {} training on teacher'.format(epoch))
        for _ in range(5):
            student.eval()
            teacher.train()
            dataloader_iterator = iter(welding_train_loader)
            try:
                sample_batched= next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(welding_train_loader)
                sample_batched = next(dataloader_iterator)
            # for i_batch, sample_batched in enumerate(welding_train_loader):
            # if (i_batch + 1) > math.ceil(training_number / batch_size):
                # break
            inputs = sample_batched['image']#.cuda()
            real_hmaps = sample_batched['hmap']#.cuda()
            empty_batch = True
            for idx, input in enumerate(inputs):
                input = input.cuda()

                pseudo_hmap = student(input.unsqueeze(0)).squeeze(0)
                G_sample = torch.cat([input, pseudo_hmap], dim=0).unsqueeze(0)

                real_hmap = real_hmaps[idx].cuda()
                real = torch.cat([input, real_hmap], dim=0).unsqueeze(0)
                if empty_batch:
                    G_sample_batch = G_sample
                    real_batch = real
                    empty_batch = False
                else:
                    G_sample_batch = torch.cat([G_sample_batch, G_sample], dim=0)
                    real_batch = torch.cat([real_batch, real]) 

            zeros_label = torch.zeros(batch_size, 1).cuda()
            ones_label = torch.ones(batch_size, 1).cuda()

            D_real = teacher(real_batch)
            D_fake = teacher(G_sample_batch)
            
            # D_loss_real = nn.functional.binary_cross_entropy(D_real, ones_label)
            # D_loss_fake = nn.functional.binary_cross_entropy(D_fake, zeros_label)
            # D_loss = D_loss_real + D_loss_fake# + alpha * student_loss

       
            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

            D_loss.backward(retain_graph=True)
            # D_loss.backward()
            D_solver.step()

            # Weight clipping
            for p in teacher.parameters():
                p.data.clamp_(-0.01, 0.01)

            # D_loss.backward(retain_graph=True)
            # D_solver.step()

            G_solver.zero_grad()
            D_solver.zero_grad()

            # correct += (torch.max(predicted, 1)[1] == torch.max(labels, 1)[1]).sum().item()
            correct += (D_real>=0.5).sum().item() + (D_fake<0.5).sum().item()
            total += batch_size * 2

        print('Start epoch {} training on student'.format(epoch))
        student.train()
        teacher.eval()
        dataloader_iterator = iter(welding_train_loader)
        try:
            sample_batched= next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(welding_train_loader)
            sample_batched = next(dataloader_iterator)


        inputs = sample_batched['image']#.cuda()
        real_hmaps = sample_batched['hmap']#.cuda()
        empty_batch = True
        for idx, input in enumerate(inputs):
            input = input.cuda()
            pseudo_hmap = student(input.unsqueeze(0)).squeeze(0)
            real_hmap = real_hmaps[idx].cuda()
            if empty_batch:
                fake_hmaps_batch = pseudo_hmap
                real_hmaps_batch = real_hmap
                empty_batch = False
            else:
                fake_hmaps_batch = torch.cat([fake_hmaps_batch, pseudo_hmap])
                real_hmaps_batch = torch.cat([real_hmaps_batch, real_hmap])

        ones_label = torch.ones(batch_size, 1).cuda()

        D_fake = teacher(G_sample_batch)
        
        # student_loss = nn.MSELoss()(fake_hmaps_batch, real_hmaps_batch)
        # Generator forward-loss-backward-update
        # alpha = 1e6
        # G_loss = nn.functional.binary_cross_entropy(D_fake, ones_label)# + alpha * student_loss

        G_loss = -torch.mean(D_fake)

        G_loss.backward()
        G_solver.step()

        G_solver.zero_grad()
        D_solver.zero_grad()

        if (epoch+1) % 5 == 0:    # every 20 mini-batches...

            print('Train batch:\tG_loss: {:.30f} D_loss: {:.30f} acc_training_D: {:.30f}%  {}/{}'.format(
                    G_loss.item(),
                    D_loss.item(),
                    100 * correct / total,
                    correct,
                    total))


            writer_gan.add_scalar("advererial_gan_G_loss", \
                    G_loss.item(), #/ len(inputs), \
                    epoch * math.ceil(len(welding_train_loader) / batch_size) \
                    )

            writer_gan.add_scalar("advererial_gan_D_loss", \
                    D_loss.item(), #/ len(inputs), \
                    epoch * math.ceil(len(welding_train_loader) / batch_size) \
                    )


            writer_gan.add_scalar("adverserial_training_D_acc", \
                    100 * correct / total, #/ len(inputs), \
                    epoch * math.ceil(len(welding_train_loader) / batch_size) \
                    )

            # correct = 0 
            # total = 0

            batch_acc_D, batch_acc_G = eval_gan()
            print('eval gan acc_D: {:.10f} acc_G: {:.10f}'.format(batch_acc_D, batch_acc_G))

            writer_gan.add_scalar("adverserial_gan_valid_acc_D", \
                    batch_acc_D, \
                    epoch
                    )
            writer_gan.add_scalar("adverserial_gan_valid_acc_G", \
                    batch_acc_G, \
                    epoch
                    )

            # if batch_correct_rate > correct_rate:
                # correct_rate = batch_correct_rate
            dir_weight = './check_points/weights_adverserial_gan_G.pth'
            torch.save(student.state_dict(), dir_weight)
            print('model saved to ' + dir_weight)

def eval_gan():
    correct_rate = 0
    student.eval()
    teacher.eval()

    total_G_loss = 0
    correct_D = 0
    total_D = 0
    correct_G = 0
    total_G = 0
    for i_batch, sample_batched in enumerate(welding_valid_loader):
        inputs = sample_batched['image']#.cuda()
        real_hmaps = sample_batched['hmap']#.cuda()
        empty_batch = True
        for idx, input in enumerate(inputs):
            input = input.cuda()

            pseudo_hmap = student(input.unsqueeze(0)).squeeze(0)
            G_sample = torch.cat([input, pseudo_hmap], dim=0).unsqueeze(0)

            real_hmap = real_hmaps[idx].cuda()
            real = torch.cat([input, real_hmap], dim=0).unsqueeze(0)
            if empty_batch:
                G_sample_batch = G_sample
                real_batch = real
                empty_batch = False
            else:
                G_sample_batch = torch.cat([G_sample_batch, G_sample], dim=0)
                real_batch = torch.cat([real_batch, real]) 

        zeros_label = torch.zeros(len(G_sample_batch), 1).cuda()
        ones_label = torch.ones(len(G_sample_batch), 1).cuda()

        D_real = teacher(real_batch)
        D_fake = teacher(G_sample_batch)
        
        correct_D += (D_real>0.5).sum().item() + (D_fake<0.5).sum().item()
        total_D += batch_size * 2

        correct_G += (D_fake>0.5).sum().item()
        total_G += batch_size

    acc_G = correct_G / total_G
    acc_D = correct_D / total_D

    return acc_D, acc_G



train_gan()

