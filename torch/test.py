from Student import Student
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
from utils import heatmap_to_coor, accuracy


########################################    Transformed Dataset

file_to_read = './csv/tail_1000.csv'
# saved_weights = './check_points/weights_adverserial_gan_G.pth'
# saved_weights = './saved_weights_4265.pth'
saved_weights = './saved_weights_200.pth'

file_to_write = "./csv/pred_dict_200.csv"
# file_to_write = "./csv/pred_dict_200_G.csv"
# file_to_write = "./csv/pred_dict_4265.csv"
batch_size = 128

transformed_dataset = WeldingDatasetToTensor(csv_file=file_to_read, root_dir='./')

valid_loader = DataLoader(transformed_dataset, batch_size=batch_size, \
                        num_workers=6)

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
with torch.no_grad():
    total_acc_x = 0
    total_acc_y = 0
    for i, batch in enumerate(valid_loader):
        inputs = batch['image'].float().cuda()
        labels = batch['coor'].float().cuda()
        img_names = batch['img_name']
        outputs = model(inputs)
        valid_loss += criterion(outputs, labels)
        outputs = outputs.cpu().detach().numpy()
        for index, out in enumerate(outputs):
            x, y = heatmap_to_coor(out.reshape(224, 224))
            f.write(img_names[index] + ',' + \
                    str(int(x / 224 * 1280)) + ',' + str(int(y / 224 * 1024)) + '\n')


        # outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        acc_x, acc_y = accuracy(outputs, labels)
        total_acc_x += acc_x
        total_acc_y += acc_y

    print("=" * 30)
    print("total acc_x = {:.10f}".format(total_acc_x/len(valid_loader)))
    # set_trace()
    print("total acc_y = {:.10f}".format(total_acc_y/len(valid_loader)))
    print("=" * 30)

    f.close()
    valid_loss = valid_loss / len(valid_loader.dataset)
    print('valid loss {}'.format(valid_loss))

