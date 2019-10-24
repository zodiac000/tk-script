from Student import Student
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import WeldingDataset
from pdb import set_trace
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image


num_epochs = 1
batch_size = 128
########################################    Transformed Dataset


transformed_dataset = WeldingDataset(csv_file='./saved_dict.csv', root_dir='./')

random_seed = 88
split = 1000
batch_size = 4
indices = list(range(len(transformed_dataset)))

train_indices, val_indices = indices[:-split], indices[-split:]


valid_sampler = SubsetRandomSampler(val_indices)

valid_loader = DataLoader(transformed_dataset, batch_size=batch_size, \
                        num_workers=6, sampler=valid_sampler)


model = Student().cuda()
model.load_state_dict(torch.load('./vae.pth'))

criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()

# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
    # print(param_tensor, "\t", model.state_dict()[param_tensor].size())


model.eval()
# valid_loss = 0
# with torch.no_grad():
    # for i, batch in enumerate(valid_loader):
        # inputs = batch['image'].float().cuda()
        # labels = batch['coor'].float().cuda()
        # outputs = model(inputs)
        # valid_loss += criterion(outputs, labels)
    # valid_loss = valid_loss / len(valid_loader.dataset)
    # print('valid loss {}'.format(valid_loss))

transform = Compose([Resize((224, 224)), ToTensor()])
image_name = './test.jpg'
image = Image.open(image_name)
input_image = transform(image)
input_image = torch.unsqueeze(input_image, 0).cuda()
output_image = model(input_image)
output = output_image.cpu().detach().numpy()
print(output)
print(output.shape)
print(output.max())
print(np.count_nonzero(output == 1.0))
