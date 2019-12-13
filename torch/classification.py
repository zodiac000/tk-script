from Student_cls import Student_cls
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import WeldingDatasetToTensor
from pdb import set_trace
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from utils import accuracy_sum, heatmap_to_coor, crop
from torchvision.transforms import ToTensor, Resize
from PIL import Image

training_number = 6500

train_dataset = WeldingDatasetToTensor(csv_file='./csv/fail_' + str(training_number) + '.csv', \
                                        root_dir='./')
val_dataset = WeldingDatasetToTensor(csv_file='./csv/fail_1000.csv', root_dir='./')
saved_weight_dir = './check_points/saved_weights_cls_' + str(training_number) + '.pth'
tensorboard_file = 'runs/classification_' + str(training_number)


num_epochs = 30000
batch_size = 16
valid_batch_size = 128
lr = 1e-5

train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                        num_workers=4, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=valid_batch_size, \
                        num_workers=4)

writer = SummaryWriter(tensorboard_file)


model = Student_cls().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def train():
    train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                        num_workers=4, shuffle=True)
    max_acc = 0
    for epoch in range(num_epochs):
        dataloader_iterator = iter(train_loader)
        try:
            sample_batched= next(dataloader_iterator)
        except StopIteration:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                                num_workers=4, shuffle=True)
            dataloader_iterator = iter(train_loader)
            sample_batched = next(dataloader_iterator)

        inputs = sample_batched['image'].cuda()
        class_real = sample_batched['class_real'].cuda()

        optimizer.zero_grad()
        pred = model(inputs)
        loss = nn.functional.binary_cross_entropy(pred, class_real)
        loss.backward()
        optimizer.step()
        

        if (epoch+1) % 5 == 0:    # every 20 mini-batches...
            print('Train epoch: {}\tLoss: {:.30f}'.format(
                    epoch+1,
                    loss.item())) #/ len(inputs)))
            writer.add_scalar("classification_training_loss", \
                    loss.item(), #/ len(inputs), \
                    epoch  + epoch * math.ceil(len(train_loader) / batch_size) \
                    )


        if (epoch+1) % 50 == 0:    # every 20 mini-batches...
            with torch.no_grad():
                valid_loss = 0
                correct = 0
                total = 0
                for i, batch in enumerate(valid_loader):
                    inputs = batch['image'].float().cuda()
                    class_real = batch['class_real'].cuda()
                    pred = model(inputs)
                    valid_loss += nn.functional.binary_cross_entropy(pred, class_real)
                    total += class_real.size(0)
                    zeros = torch.zeros(pred.shape).cuda()
                    ones = torch.ones(pred.shape).cuda()
                    pred_binary = torch.where(pred >= 0.5, ones, zeros)
                    correct += (pred_binary == class_real).sum().item()


                valid_loss = valid_loss / len(valid_loader)
                valid_acc = correct / total 
                print('Valid loss {}'.format(valid_loss))
                writer.add_scalar("classification_Val_loss", valid_loss, epoch)

                print('Valid acc {}'.format(valid_acc))
                writer.add_scalar("classification_Val_Acc", 100 * valid_acc, epoch)
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), saved_weight_dir)
                    print('model saved to ' + saved_weight_dir)


if __name__ == '__main__':
    train()
