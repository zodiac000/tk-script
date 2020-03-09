from Student_bc import Student
# from Student2 import Student2
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
from utils import heatmap_to_coor, accuracy_sum, get_total_confidence
from tqdm import tqdm
########################################    Transformed Dataset

# file_to_read = './csv/all.csv'
# file_to_write = "./csv/pred_all.csv"


file_to_read = './csv/pass_valid_head_6415.csv'
file_to_write = "./csv/pred_pass_valid_head_6415.csv"



# saved_weights = './check_points/weights_800.pth'
saved_weights = './check_points/weights_1000+79.pth'



batch_size = 4

dataset = WeldingDatasetToTensor(csv_file=file_to_read, root_dir='./')


valid_loader = DataLoader(dataset, batch_size=batch_size, \
                        num_workers=1)

model = Student().cuda()
model.load_state_dict(torch.load(saved_weights))

criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()

# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
    # print(param_tensor, "\t", model.state_dict()[param_tensor].size())


model.eval()
valid_loss = 0
f = open(file_to_write, "w")
all_acc_x = []
all_acc_y = []
pbar = tqdm(total=len(valid_loader.dataset))
with torch.no_grad():
    total_acc_x = 0
    total_acc_y = 0
    e_distances = 0
    distances = []
    for i, batch in enumerate(valid_loader):
        inputs = batch['image'].float().cuda()
        labels = batch['hmap'].float().cuda()
        coors = batch['coor_bc'].numpy()
        img_names = batch['img_name']
        outputs = model(inputs)
        valid_loss += criterion(outputs, labels)
        outputs = outputs.cpu().detach().numpy()
        for index, out in enumerate(outputs):
            x, y = heatmap_to_coor(out.reshape(224, 224))
            total_conf = get_total_confidence(out.reshape(224,224))
            # mean_conf = np.asarray(total_conf).mean()
            total_conf = ','.join(map(str, total_conf))

            e_distance = np.sum((np.asarray([x, y]) * [1280/224, 1024/224] - coors[index]) ** 2, axis=0) ** 0.5

            # e_distance = ((int(x/224*1280)-coors[index][0])**2 + \
                            # (int(y/224*1024)-coors[index][1])**2)**0.5
            # acc_x = abs(int(x/224*1280) - coors[index][0]) / 1280
            # acc_y = abs(int(y/224*1024) - coors[index][1]) / 1024
            f.write(img_names[index]\
                    # + ',' + str(coors[index][0]) \
                    # + ',' + str(coors[index][1]) \
                    + ',' + str(int(x / 224 * 1280)) \
                    + ',' + str(int(y / 224 * 1024))\
                    # + ',' + str(acc_x)\
                    # + ',' + str(acc_y)\
                    # + ',' + str(out.max()) \
                    + ',' + str(e_distance)   \
                    # + ',' + str(mean_conf)  \
                    # + ',' + str(total_conf)\
                    + '\n')
            # print("wrtie to file")
                    
            distances.append(e_distance)
            # e_distances += e_distance
            pbar.update()

        # outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        sum_acc_x, sum_acc_y, list_acc_x, list_acc_y = accuracy_sum(outputs, coors)
        total_acc_x += sum_acc_x
        total_acc_y += sum_acc_y
        all_acc_x.extend(list_acc_x)
        all_acc_y.extend(list_acc_y)
        e_distances = np.asarray(distances)

    print("=" * 30)
    print("total acc_x = {:.10f}".format(total_acc_x/len(valid_loader.dataset)))
    print("total acc_y = {:.10f}".format(total_acc_y/len(valid_loader.dataset)))
    print("Euclidean Distance: {}".format(np.mean(e_distances)))
    print("=" * 30)

    f.close()
    valid_loss = valid_loss / len(valid_loader.dataset)
    print('valid loss {}'.format(valid_loss))



