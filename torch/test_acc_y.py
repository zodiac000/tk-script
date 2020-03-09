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

file_to_read = './csv/pass_valid_head_6415.csv'
file_to_write = "./csv/pred_pass_valid_head_6415_acc_y.csv"

# file_to_read = './csv/pass_valid_6000.csv'
# file_to_write = "./csv/pred_pass_valid_6000_acc.csv"

# file_to_read = './csv/pass_valid_tail_800.csv'
# file_to_write = "./csv/pred_pass_valid_tail_800_acc.csv"

saved_weights_s = './check_points/weights_1000.pth'
saved_weights_t = './check_points/weights_train_acc_y.pth'


batch_size = 20

dataset = WeldingDatasetToTensor(csv_file=file_to_read, root_dir='./')
# dataset_label = WeldingDatasetToTensor(csv_file=file_to_read_label, root_dir='./')


valid_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
# valid_loader_label = DataLoader(dataset_label, batch_size=batch_size, num_workers=1)

student = Student().cuda()
teacher = Teacher().cuda()
student.load_state_dict(torch.load(saved_weights_s))
teacher.load_state_dict(torch.load(saved_weights_t))
student.eval()
teacher.eval()

# criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()



# valid_loss = 0
positive = 0
correct_real = 0
correct_fake = 0
total = 0 
pbar = tqdm(total=len(valid_loader.dataset))
all_distances = []
with open(file_to_write, "w") as f:
    with torch.no_grad():
        # temp = iter(valid_loader_label)
        for i, batch in enumerate(valid_loader):
            # label_batch = next(temp)
            # coors_gt = label_batch['coor_bc'].numpy()

            inputs = batch['image'].float().cuda()
            real_hmaps = batch['hmap'].float().cuda()
            coors_gt = batch['coor_bc'].numpy()
            img_names = batch['img_name']

            outputs = student(inputs)
            coors_pred = []
            for out in outputs:
                coors_pred.append(list(heatmap_to_coor(out.detach().cpu().numpy())))
            coors_pred = np.asarray(coors_pred)
            coors_pred = coors_pred * [1280/224, 1024/224]
            coors_pred = np.asarray(coors_pred).astype(int)
            distances = np.sum((coors_gt - coors_pred) ** 2, axis=1) ** 0.5
            distances = np.asarray(distances).astype(float)
            all_distances.append(distances)
            
            empty_batch = True
            for idx, input in enumerate(inputs):
                input = input.cuda()

                # pseudo_hmap = student(input.unsqueeze(0)).squeeze(0)
                pseudo_hmap = outputs[idx]
                # pseudo_hmap = spike(pseudo_hmap)
                pseudo = torch.cat([input, pseudo_hmap.cuda()], dim=0).unsqueeze(0)

                # real_hmap = spike(real_hmaps[idx]).cuda()
                # real = torch.cat([input.float(), real_hmap.float()], dim=0).unsqueeze(0)
                if empty_batch:
                    pseudo_batch = pseudo
                    # real_batch = real
                    empty_batch = False
                else:
                    pseudo_batch = torch.cat([pseudo_batch, pseudo], dim=0)
                    # real_batch = torch.cat([real_batch, real]) 

                pbar.update()

            # zeros_label = torch.zeros(batch_size, 1).cuda()
            # ones_label = torch.ones(batch_size, 1).cuda()

            # D_real = teacher(real_batch)
            D_fake = teacher(pseudo_batch)
            # positive += (D_real>=0.5).sum().item() 
            # correct += (D_real>=0.5).sum().item() + (D_fake < 0.5).sum().item()
            # correct_real += (D_real>=0.5).sum().item() 
            # correct_fake += (D_fake < 0.5).sum().item()
            # total += batch_size 


            for index, conf in enumerate(D_fake):
                conf = conf.detach().cpu().numpy()
                f.write(
                        img_names[index] + \
                        ',' + str(conf[0]) + \
                        # ',' + str(conf[1]) + \
                        ',' + str(distances[index]) + \
                        ',' + str(coors_pred[index][0]) + \
                        ',' + str(coors_pred[index][1]) + \
                        '\n' 
                        )


    all_distances = np.asarray(all_distances)

    all_distances = np.append(np.array(list(all_distances[:-1])).flatten(), all_distances[-1]) 

    # print('positive / total: {} / {} ==== {}%'.format(positive, total, positive/total * 100))
    # print('correct_real / total: {} / {} ==== {}%'.format(correct_real, total,\
            # correct_real/total * 100))
    # print('correct_fake / total: {} / {} ==== {}%'.format(correct_fake, total, \
            # correct_fake/total * 100))
    print('total number of samples: {}'.format(all_distances.shape[0]))
    print('mean distance: {}'.format(all_distances.mean()))



