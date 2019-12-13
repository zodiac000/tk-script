from Student_cls import Student_cls
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
import matplotlib.pyplot as plt
########################################    Transformed Dataset

# file_to_read = './csv/fail_1000.csv'
file_to_read = './csv/pass_images_7500.csv'
saved_weights = './check_points/saved_weights_cls_6500.pth'
file_to_write = "./csv/pred_cls_6500.csv"
batch_size = 128
dataset = WeldingDatasetToTensor(csv_file=file_to_read, root_dir='./')
valid_loader = DataLoader(dataset, batch_size=batch_size, \
                        num_workers=1)

model = Student_cls().cuda()
model.load_state_dict(torch.load(saved_weights))


# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
    # print(param_tensor, "\t", model.state_dict()[param_tensor].size())


f = open(file_to_write, "w")
with torch.no_grad():
    loss = 0 
    correct = 0
    total = 0

    for i, batch in enumerate(valid_loader):
        inputs = batch['image'].cuda()
        class_real = batch['class_real'].cuda()
        img_names = batch['img_name']
        preds = model(inputs)
        loss += nn.functional.binary_cross_entropy(preds, class_real)
        total += class_real.size(0)
        zeros = torch.zeros(preds.shape).cuda()
        ones = torch.ones(preds.shape).cuda()
        preds_binary = torch.where(preds >= 0.5, ones, zeros)
        correct += (preds_binary == class_real).sum().item()
        # set_trace()
        for index, pred in enumerate(preds_binary):
            # f.write(img_names[index] + ',' + class_real[index], + ',' + pred + '\n')
            f.write(','.join(list((img_names[index],\
                                    str(class_real[index].item()),\
                                    str(pred.item())))) + '\n')
    loss = loss / len(valid_loader)
    acc = correct / total

    f.close()
    print('valid loss {}'.format(loss))
    print('acc {}'.format(acc))



