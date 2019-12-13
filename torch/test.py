from Student_reg import Student
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
from utils import heatmap_to_coor, accuracy_sum
########################################    Transformed Dataset

file_to_read = './csv/pass_adbc_1000.csv'
# file_to_read = './csv/failed.csv'

# saved_weights = './check_points/saved_weights_50.pth'
# file_to_write = "./csv/pred_dict_50.csv"

# saved_weights = './check_points/saved_weights_200.pth'
# file_to_write = "./csv/pred_dict_200.csv"

# saved_weights = './check_points/weights_adverserial_gan_G.pth'
# file_to_write = "./csv/pred_dict_200_G.csv"

# saved_weights = './saved_weights_4265.pth'
# file_to_write = "./csv/pred_dict_4265.csv"

# saved_weights = './check_points/saved_weights_cascade_4265.pth'
# file_to_write = "./csv/pred_dict_cascade_4265.csv"

# saved_weights = './check_points/saved_weights_cascade2_4265.pth'
# file_to_write = "./csv/pred_dict_cascade2_4265.csv"

# saved_weights = './check_points/saved_weights_cascade4_4265.pth'
# file_to_write = "./csv/pred_dict_cascade4_4265.csv"


# saved_weights = './check_points/saved_weights_cascade4_6415.pth'
# file_to_write = "./csv/pred_dict_cascade4_6415.csv"

saved_weights = './check_points/saved_weights_ad_6500.pth'
# file_to_write = "./csv/pred_dict_cascade4_6500.csv"
file_to_write = "./csv/pred_ad_6500.csv"
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
            f.write(img_names[index] + ',' + \
                    str(int(x / 224 * 1280)) + ',' + str(int(y / 224 * 1024)) + '\n')
            e_distance = ((int(x/224*1280)-coors[index][0])**2 + \
                            (int(y/224*1024)-coors[index][1])**2)**0.5
            distances.append(e_distance)
            e_distances += e_distance

        # outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        sum_acc_x, sum_acc_y, list_acc_x, list_acc_y = accuracy_sum(outputs, coors)
        total_acc_x += sum_acc_x
        total_acc_y += sum_acc_y
        all_acc_x.extend(list_acc_x)
        all_acc_y.extend(list_acc_y)

    print("=" * 30)
    print("total acc_x = {:.10f}".format(total_acc_x/len(valid_loader.dataset)))
    print("total acc_y = {:.10f}".format(total_acc_y/len(valid_loader.dataset)))
    print("Euclidean Distance: {}".format(e_distances/len(valid_loader.dataset)))
    print("=" * 30)

    f.close()
    valid_loss = valid_loss / len(valid_loader.dataset)
    print('valid loss {}'.format(valid_loss))



