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
train_csv = './csv/head_' + str(training_number) + '.csv'
val_csv = './csv/tail_1000.csv'

weight_to_load_student = './check_points/saved_weights_' + str(training_number) + '_39.pth'
writer_teacher_dir = 'runs/adverserial_teacher_' + str(training_number)


weight_to_load_teacher = './check_points/weights_adverserial_teacher_39.pth'
writer_student_dir = 'runs/adverserial_student_' + str(training_number)


writer_gan_dir = 'runs/adverserial_gan_' + str(training_number)

student = Student().cuda()
student.load_state_dict(torch.load(weight_to_load_student))

teacher = Teacher().cuda()
# teacher.load_state_dict(torch.load(weight_to_load_teacher))

train_dataset = WeldingDatasetToTensor(csv_file=train_csv, root_dir='./')
valid_dataset = WeldingDatasetToTensor(csv_file=val_csv, root_dir='./')

num_epochs = 100
batch_size = 4
batch_size_valid = 4
lr = 1e-6

welding_train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                        num_workers=1, shuffle=True)
welding_valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, \
                        num_workers=1, shuffle=True)

writer_teacher = SummaryWriter(writer_teacher_dir)
writer_student = SummaryWriter(writer_student_dir)
writer_gan = SummaryWriter(writer_gan_dir)


# criterion = nn.MSELoss().cuda()
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=lr)
optimizer_student = torch.optim.Adam(student.parameters(), lr=lr)

G_solver = torch.optim.Adam(student.parameters(), lr=lr)
D_solver = torch.optim.Adam(teacher.parameters(), lr=lr)
                        
def train_gan():
    correct_rate = 0
    for epoch in range(num_epochs):
        student.train()
        teacher.train()
        welding_train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                        num_workers=1, shuffle=True)
        total_G_loss = 0
        total_D_loss = 0
        correct = 0
        total = 0
        print('Start epoch {} training on teacher'.format(epoch))
        for i_batch, sample_batched in enumerate(welding_train_loader):
            inputs = sample_batched['image']#.cuda()
            real_hmaps = sample_batched['coor']#.cuda()
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
                    fake_hmaps_batch = pseudo_hmap
                    real_hmaps_batch = real_hmap
                    empty_batch = False
                else:
                    G_sample_batch = torch.cat([G_sample_batch, G_sample], dim=0)
                    real_batch = torch.cat([real_batch, real]) 
                    fake_hmaps_batch = torch.cat([fake_hmaps_batch, pseudo_hmap])
                    real_hmaps_batch = torch.cat([real_hmaps_batch, real_hmap])

            zeros_label = torch.zeros(batch_size, 1).cuda()
            ones_label = torch.ones(batch_size, 1).cuda()

            D_real = teacher(real_batch)
            D_fake = teacher(G_sample_batch)
            
            student_loss = nn.MSELoss()(fake_hmaps_batch, real_hmaps_batch)
            alpha = 1e6
            D_loss_real = nn.functional.binary_cross_entropy(D_real, ones_label)
            D_loss_fake = nn.functional.binary_cross_entropy(D_fake, zeros_label)
            D_loss = D_loss_real + D_loss_fake# + alpha * student_loss

            D_loss.backward(retain_graph=True)
            # student_loss.backward()
            D_solver.step()
            # G_solver.step()

            # Housekeeping - reset gradient
            # reset_grad()
            G_solver.zero_grad()
            D_solver.zero_grad()

            # Generator forward-loss-backward-update
            G_loss = nn.functional.binary_cross_entropy(D_fake, ones_label) + alpha * student_loss

            G_loss.backward()
            G_solver.step()

            # Housekeeping - reset gradient
            # reset_grad()
            G_solver.zero_grad()
            D_solver.zero_grad()

            # correct += (torch.max(predicted, 1)[1] == torch.max(labels, 1)[1]).sum().item()
            correct += (D_real>0.5).sum().item() + (D_fake<0.5).sum().item()
            total += batch_size * 2

            total_G_loss += G_loss.item()
            total_D_loss += D_loss.item()
            if i_batch % 4 == 0:    # every 20 mini-batches...

                print('Train batch: {} [{}/{} ({:.0f}%)]\tG_loss: {:.30f} D_loss: {:.30f} accuray: {:.30f}%  {}/{}'.format(
                        i_batch,
                        i_batch * len(inputs),
                        len(welding_train_loader.dataset), 100. * i_batch / len(welding_train_loader),
                        G_loss.item(),
                        D_loss.item(),
                        100 * correct / total,
                        correct,
                        total))


                writer_gan.add_scalar("advererial_gan_G_loss", \
                        G_loss.item(), #/ len(inputs), \
                        i_batch  + epoch * math.ceil(len(welding_train_loader) / batch_size) \
                        )

                writer_gan.add_scalar("advererial_gan_D_loss", \
                        D_loss.item(), #/ len(inputs), \
                        i_batch  + epoch * math.ceil(len(welding_train_loader) / batch_size) \
                        )


                writer_gan.add_scalar("adverserial_teacher_training_acc", \
                        100 * correct / total, #/ len(inputs), \
                        i_batch  + epoch * math.ceil(len(welding_train_loader) / batch_size) \
                        )

                correct = 0 
                total = 0
        # print('epoch [{}/{}], training loss:{:.30f}'.format(epoch+1, num_epochs, \
                                         # train_loss / len(welding_train_loader)))

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
    total_D_loss = 0
    correct_D = 0
    total_D = 0
    correct_G = 0
    total_G = 0
    for i_batch, sample_batched in enumerate(welding_valid_loader):
        inputs = sample_batched['image']#.cuda()
        real_hmaps = sample_batched['coor']#.cuda()
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
        # D_loss = D_loss_real + D_loss_fake

        # Generator forward-loss-backward-update
        # G_loss = nn.functional.binary_cross_entropy(D_fake, ones_label)

        # correct += (torch.max(predicted, 1)[1] == torch.max(labels, 1)[1]).sum().item()
        correct_D += (D_real>0.5).sum().item() + (D_fake<0.5).sum().item()
        total_D += batch_size * 2

        correct_G += (D_fake>0.5).sum().item()
        total_G += batch_size

        # total_G_loss += G_loss.item()
        # total_D_loss += D_loss.item()

    acc_G = correct_G / total_G
    acc_D = correct_D / total_D

    return acc_D, acc_G


def train_teacher(num_epochs):
    student.eval()
    correct_rate = 0
    for epoch in range(num_epochs):
        teacher.train()
        train_loss = 0
        valid_loss = 0
        correct = 0
        total = 0
        print('Start epoch {} training on teacher'.format(epoch))
        for i_batch, sample_batched in enumerate(welding_train_loader):
            inputs = sample_batched['image']#.cuda()
            real_hmaps = sample_batched['coor']#.cuda()
            optimizer_teacher.zero_grad()
            empty_batch = True
            # labels = None
            for idx, input in enumerate(inputs):
                input = input.cuda()
                if np.random.uniform() < 0.5:
                    pseudo_hmap = student(input.unsqueeze(0)).squeeze(0)
                    concats = torch.cat([input, pseudo_hmap], dim=0).unsqueeze(0)
                    # concats = pseudo_hmap.unsqueeze(0)
                    # label = torch.zeros(2)
                    label = torch.empty(2)
                    label[0].uniform_(0, 0.1)
                    label[1].uniform_(0.9, 1)
                    # label[1] = 1
                    # label[0] = uniform(0, 0.1) #flipped soft noisy label
                    # label[1] = uniform(0.9, 1) #flipped soft noisy label
                    label = label.cuda().unsqueeze(0)
                    # print("from student")

                else:
                    real_hmap = real_hmaps[idx].cuda()
                    # concats = real_hmap.unsqueeze(0)
                    concats = torch.cat([input, real_hmap], dim=0).unsqueeze(0)
                    # label = torch.zeros(2).cuda().unsqueeze(0)
                    label = torch.empty(2)
                    label[1].uniform_(0, 0.1)
                    label[0].uniform_(0.9, 1)
                    # label = torch.zeros(2)
                    # label[0] = 1
                    # label[1] = uniform(0, 0.1) #flipped soft noisy label
                    # label[0] = uniform(0.9, 1) #flipped soft noisy label
                    label = label.cuda().unsqueeze(0)
                    # print("from true label")
                if empty_batch:
                    concatenated_batch = concats
                    labels = label
                    empty_batch = False
                else:
                    concatenated_batch = torch.cat([concatenated_batch, concats], dim=0)
                    labels = torch.cat([labels, label]) 
            ########save batch images #########################
            # for i, images in enumerate(concatenated_batch):
                # input = images[0].cpu().detach().numpy()
                # hmap = images[1].cpu().detach().numpy()
                # img_input = Image.fromarray(np.uint8(input * 255))
                # img_hmap = Image.fromarray(np.uint8(hmap * 255))
                # img_input.save("input_" + str(i) + '.jpg')
                # img_hmap.save("hmap_" + str(i) + '.jpg')
            ######################################################
            predicted = teacher(concatenated_batch)
            loss = criterion(predicted, torch.max(labels, 1)[1])
            # loss = criterion(predicted, labels)
            # norm_predicted = 1/(1+np.exp(-predicted))
            # correct += (predicted == labels).sum().item()
            correct += (torch.max(predicted, 1)[1] == torch.max(labels, 1)[1]).sum().item()
            # set_trace()
            total += labels.size(0)
            loss.backward()
            optimizer_teacher.step()

            train_loss += loss.item()
            if i_batch % 4 == 0:    # every 20 mini-batches...

                print('Train batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.30f} accuray: {:.30f}%  {}/{}'.format(
                        i_batch,
                        i_batch * len(inputs),
                        len(welding_train_loader.dataset), 100. * i_batch / len(welding_train_loader),
                        loss.item(),
                        100 * correct / total,
                        correct,
                        total))


                writer_teacher.add_scalar("adverserial_teacher_training_loss", \
                        loss.item(), #/ len(inputs), \
                        i_batch  + epoch * math.ceil(len(welding_train_loader) / batch_size) \
                        )

                writer_teacher.add_scalar("adverserial_teacher_training_acc", \
                        100 * correct / total, #/ len(inputs), \
                        i_batch  + epoch * math.ceil(len(welding_train_loader) / batch_size) \
                        )

                correct = 0 
                total = 0
        print('epoch [{}/{}], training loss:{:.30f}'.format(epoch+1, num_epochs, \
                                         train_loss / len(welding_train_loader)))

        batch_correct_rate = eval_teacher()

        writer_teacher.add_scalar("adverserial_teacher_valid_acc", \
                batch_correct_rate, #/ len(inputs), \
                epoch
                )

        if batch_correct_rate > correct_rate:
            correct_rate = batch_correct_rate
            dir_weight = './check_points/weights_adverserial_teacher.pth'
            torch.save(teacher.state_dict(), dir_weight)
            print('model saved to ' + dir_weight)
######################################################################
def eval_teacher(weights_teacher=None):
    teacher.eval()
    student.eval()
    if weights_teacher:
        teacher.load_state_dict(torch.load(weights_teacher))

    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(welding_valid_loader):
            inputs = sample_batched['image']#.cuda()
            real_hmaps = sample_batched['coor']#.cuda()
            empty_batch = True

            for idx, input in enumerate(inputs):
                input = input.cuda()
                if np.random.uniform() < 0.5:
                    pseudo_hmap = student(input.unsqueeze(0)).squeeze(0)
                    concats = torch.cat([input, pseudo_hmap], dim=0).unsqueeze(0)
                    label = torch.zeros(2)
                    label[1] = 1
                    label = label.cuda().unsqueeze(0).long()
                else:
                    real_hmap = real_hmaps[idx].cuda()
                    concats = torch.cat([input, real_hmap], dim=0).unsqueeze(0)
                    label = torch.zeros(2)
                    label[0] = 1
                    label = label.cuda().unsqueeze(0).long()
                if empty_batch:
                    concatenated_batch = concats
                    labels = label
                    empty_batch = False
                else:
                    concatenated_batch = torch.cat([concatenated_batch, concats], dim=0)
                    labels = torch.cat([labels, label]) 

            predicted = teacher(concatenated_batch)
            loss = criterion(predicted, torch.max(labels, 1)[1])
            batch_correct = (torch.max(predicted, 1)[1] == torch.max(labels, 1)[1]).sum().item()
            correct += batch_correct
            total += labels.size(0)
            valid_loss += loss.item()
            
            # print('correct / total: {} / {}'.format(batch_correct, labels.size(0)))
        # print('validation loss:{:.30f}'.format( \
                                         # valid_loss / len(welding_valid_loader)))

        print('correct / total: {} / {}'.format(correct, total))
        acc = correct / total
    return acc

def train_student(num_epochs):
    teacher.eval()
    student.train()
    for epoch in range(num_epochs):
        train_loss = 0
        valid_loss = 0
        correct = 0
        total = 0
        print('Start epoch {} training on student'.format(epoch))
        for i_batch, sample_batched in enumerate(welding_valid_loader):
            inputs = sample_batched['image']#.cuda()
            real_hmaps = sample_batched['coor']#.cuda()
            optimizer_student.zero_grad()
            empty_batch = True
            # labels = None
            for idx, input in enumerate(inputs):
                input = input.cuda()

                pseudo_hmap = student(input.unsqueeze(0)).squeeze(0)
                concats = torch.cat([input, pseudo_hmap], dim=0).unsqueeze(0)
                # concats = pseudo_hmap.unsqueeze(0)
                label = torch.zeros(2)
                label[0] = 1
                label = label.cuda().unsqueeze(0).long()
                # print("from student")

                if empty_batch:
                    concatenated_batch = concats
                    labels = label
                    empty_batch = False
                else:
                    concatenated_batch = torch.cat([concatenated_batch, concats], dim=0)
                    labels = torch.cat([labels, label]) 
            
            predicted = teacher(concatenated_batch)
            loss = criterion(predicted, torch.max(labels, 1)[1])
            correct += (torch.max(predicted, 1)[1] == torch.max(labels, 1)[1]).sum().item()
            total += labels.size(0)
            loss.backward()
            optimizer_student.step()

            train_loss += loss.item()
            if i_batch % 20 == 0:    # every 20 mini-batches...

                print('Train batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.30f} accuray: {:.30f}%  {}/{}'.format(
                        i_batch,
                        i_batch * len(inputs),
                        len(welding_train_loader.dataset), 100. * i_batch / len(welding_train_loader),
                        loss.item(),
                        100 * correct / total,
                        correct,
                        total))


                writer_student.add_scalar("adverserial_student_training_loss", \
                        loss.item(), #/ len(inputs), \
                        i_batch  + epoch * math.ceil(len(welding_train_loader) / batch_size) \
                        )

                writer_student.add_scalar("adverserial_student_training_acc", \
                        100 * correct / total, #/ len(inputs), \
                        i_batch  + epoch * math.ceil(len(welding_train_loader) / batch_size) \
                        )

                correct = 0 
                total = 0
        print('epoch [{}/{}], training loss:{:.30f}'.format(epoch+1, num_epochs, \
                                         train_loss / len(welding_train_loader)))

        dir_weight = './check_points/weights_adverserial_student_{}.pth'
        torch.save(teacher.state_dict(), dir_weight.format(str(epoch)))
        print('model saved to ' + dir_weight.format(str(epoch)))

train_gan()
# train_teacher(num_epochs)
# train_student(num_epochs)
# eval_teacher(weight_to_load_teacher)

