from Student_bc import Student
from Teacher import Teacher
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import WeldingDatasetToTensor
from pdb import set_trace
import numpy as np
from numpy import unravel_index
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
import matplotlib.pyplot as plt
from utils import heatmap_to_coor, accuracy_sum, get_total_confidence, spike, gaussion_hmap
from tqdm import tqdm
########################################    Transformed Dataset

file_to_read = './csv/pass_valid_head_6415.csv'
file_to_write = "./csv/pred_pass_valid_head_6415_adv.csv"

# file_to_read = './csv/pass_valid_6000.csv'
# file_to_write = "./csv/pred_pass_valid_6000_adv.csv"

saved_weights_s = './check_points/weights_1000.pth'
# saved_weights_t = './check_points/weights_adv_mse.pth'
saved_weights_t = './check_points/weights_adv_bce.pth'


batch_size = 20
dist_lower_bound = 10.0
dataset = WeldingDatasetToTensor(csv_file=file_to_read, root_dir='./',\
                                 dist_lower_bound=dist_lower_bound)
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

pass_distances = []
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
            coors = []
            for out in outputs:
                coors.append(list(heatmap_to_coor(out.detach().cpu().numpy())))
            coors = np.asarray(coors)
            coors = coors * [1280/224, 1024/224]
            coors = np.asarray(coors).astype(int)
            distances = np.sum((coors_gt - coors) ** 2, axis=1) ** 0.5
            distances = np.asarray(distances)
            empty_batch = True
            for idx, input in enumerate(inputs):
                input = input.cuda()

                # pseudo_hmap = student(input.unsqueeze(0)).squeeze(0)
                pseudo_hmap = outputs[idx].detach().cpu().numpy()
                # pseudo_hmap = spike(pseudo_hmap)
                _, y, x = np.unravel_index(pseudo_hmap.argmax(), pseudo_hmap.shape)
                pseudo_hmap = gaussion_hmap(x, y)
                pseudo_hmap = torch.from_numpy(pseudo_hmap)
                pseudo_hmap = nn.Softmax(1)(pseudo_hmap.view(-1, 224*224)).view(1, 224, 224)

                pseudo = torch.cat([input, pseudo_hmap.cuda().float()], dim=0).unsqueeze(0)

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
                # if conf.item() >= 0.6:
                    f.write(
                            img_names[index] + \
                            # ',' + str(coors[index][0]) + \
                            # ',' + str(coors[index][1]) + \
                            ',' + str(conf.item()) + \
                            ',' + str(distances[index]) + \
                            '\n' 
                            )
                    pass_distances.append(float(distances[index]))

    pass_distances = np.asarray(pass_distances)

    pass_distances = np.append(np.array(list(pass_distances[:-1])).flatten(), pass_distances[-1]) 

    # print('positive / total: {} / {} ==== {}%'.format(positive, total, positive/total * 100))
    # print('correct_real / total: {} / {} ==== {}%'.format(correct_real, total,\
            # correct_real/total * 100))
    # print('correct_fake / total: {} / {} ==== {}%'.format(correct_fake, total, \
            # correct_fake/total * 100))
    print('The number of passed: {} mean distances is {}'.format(len(pass_distances), \
            pass_distances.mean()))


