from Student_bc import Student
from Teacher import Teacher
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import WeldingDatasetToTensor
from pdb import set_trace
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
import matplotlib.pyplot as plt
from utils import heatmap_to_coor, accuracy_sum, get_total_confidence, spike
from tqdm import tqdm
########################################    Transformed Dataset

file_to_read = './csv/pass_valid_5000_unlabel.csv'
file_to_write = "./csv/pred_pass_valid_5000_unlabel.csv"


saved_weights_s = './check_points/saved_weights_415_label.pth'
saved_weights_t = './check_points/weights_adverserial_gan_D.pth'


batch_size = 20

dataset = WeldingDatasetToTensor(csv_file=file_to_read, root_dir='./')


valid_loader = DataLoader(dataset, batch_size=batch_size, \
                        num_workers=1)

student = Student().cuda()
teacher = Teacher().cuda()
student.load_state_dict(torch.load(saved_weights_s))
teacher.load_state_dict(torch.load(saved_weights_t))
student.eval()
teacher.eval()

# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()



valid_loss = 0
correct = 0
total = 0 
pbar = tqdm(total=len(valid_loader.dataset))

with open(file_to_write, "w") as f:
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            inputs = batch['image'].float().cuda()
            real_hmaps = batch['hmap'].float().cuda()
            coors = batch['coor_bc'].numpy()
            img_names = batch['img_name']

            empty_batch = True
            for idx, input in enumerate(inputs):
                input = input.cuda()

                pseudo_hmap = student(input.unsqueeze(0)).squeeze(0)
                pseudo_hmap = spike(pseudo_hmap).cuda()
                pseudo = torch.cat([input, pseudo_hmap], dim=0).unsqueeze(0)

                real_hmap = spike(real_hmaps[idx]).cuda()
                real = torch.cat([input.float(), real_hmap.float()], dim=0).unsqueeze(0)
                if empty_batch:
                    pseudo_batch = pseudo
                    real_batch = real
                    empty_batch = False
                else:
                    pseudo_batch = torch.cat([pseudo_batch, pseudo], dim=0)
                    real_batch = torch.cat([real_batch, real]) 

                pbar.update()

            zeros_label = torch.zeros(batch_size, 1).cuda()
            ones_label = torch.ones(batch_size, 1).cuda()

            D_real = teacher(real_batch)
            D_fake = teacher(pseudo_batch)
            
            D_loss_real = nn.functional.binary_cross_entropy(D_real, ones_label)
            D_loss_fake = nn.functional.binary_cross_entropy(D_fake, zeros_label)
            D_loss = D_loss_real + D_loss_fake# + alpha * student_loss
            valid_loss += D_loss.item()

        
            # correct += (torch.max(predicted, 1)[1] == torch.max(labels, 1)[1]).sum().item()
            correct += (D_real>=0.6).sum().item() + (D_fake<0.4).sum().item()
            total += batch_size * 2




    valid_loss = valid_loss / len(valid_loader.dataset)

    print('valid loss {}'.format(valid_loss))
    print('acc_valid: {} / {} ==== {}%'.format(correct, total, correct/total * 100))




