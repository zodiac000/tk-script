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

# file_to_read = './csv/1000.csv'
# file_to_write = "./csv/1000pred.csv"

saved_weights = './check_points/weights_50_6415.pth'

batch_size = 4

dataset = WeldingDatasetToTensor(csv_file=file_to_read, data_root='all_images')


valid_loader = DataLoader(dataset, batch_size=batch_size, \
                        num_workers=1)

model = Student().cuda()
model.load_state_dict(torch.load(saved_weights))

criterion = nn.MSELoss()

model.eval()
valid_loss = 0
f = open(file_to_write, "w")
pbar = tqdm(total=len(valid_loader.dataset))
with torch.no_grad():
    distances = []
    accuracy = []
    for i, batch in enumerate(valid_loader):
        inputs = batch['image'].float().cuda()
        labels = batch['hmap'].float().cuda()
        coors = batch['coor_1'].numpy()
        img_names = batch['img_name']
        # origin_img = batch['origin_img']
        # origin_shape = np.array(origin_img[0].cpu().detach().numpy().shape)
        outputs = model(inputs)
        valid_loss += criterion(outputs, labels)
        outputs = outputs.cpu().detach().numpy()


        for index, out in enumerate(outputs):
            coor_pred = np.array(heatmap_to_coor(out.squeeze()))
            coor_pred = (coor_pred * [1280/224, 1024/224]).astype(int)
            coor_real = coors[index]
            dist = np.sum((coor_pred - coor_real) ** 2) ** 0.5
            acc = ([1, 1] - (np.absolute(coor_pred - coor_real) / [1280, 1024])) * 100

            f.write(img_names[index]\
                    + ',' + str(coor_real[0]) \
                    + ',' + str(coor_real[1]) \
                    + ',' + str(coor_pred[0]) \
                    + ',' + str(coor_pred[1])\
                    + '\n')
            # print("wrtie to file")
                    
            distances.append(dist)
            accuracy.append(acc)

            pbar.update()

        
    e_distances = np.asarray(distances)
    accuracy = np.asarray(accuracy)
    accuracy = np.mean(accuracy, axis=0)

    

    print("=" * 30)
    print("mean acc_x = {:.10f}".format(accuracy[0]))
    print("mean acc_y = {:.10f}".format(accuracy[1]))
    print("Euclidean Distance: {}".format(np.mean(e_distances)))
    print("=" * 30)

    f.close()
    valid_loss = valid_loss / len(valid_loader.dataset)
    print('valid loss {}'.format(valid_loss))



