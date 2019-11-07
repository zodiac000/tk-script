from Student import Student
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import WeldingDatasetToTensor
from pdb import set_trace
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from utils import accuracy

training_number = 200

train_dataset = WeldingDatasetToTensor(csv_file='./csv/head_' + str(training_number) + '.csv', \
                                        root_dir='./')
val_dataset = WeldingDatasetToTensor(csv_file='./csv/tail_1000.csv', root_dir='./')
saved_weight_dir = './check_points/saved_weights_' + str(training_number) + '_{}.pth'
tensorboard_file = 'runs/bootstrap_' + str(training_number)

for i in range(len(train_dataset)):
    sample = train_dataset[i]
    print(i, sample['image'].size(), sample['coor'].size())
    if i == 3:
        break

num_epochs = 40
batch_size = 32
lr = 1e-3

train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                        num_workers=6, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, \
                        num_workers=6)

writer = SummaryWriter(tensorboard_file)


model = Student().cuda()
criterion = nn.MSELoss().cuda()
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train():
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        valid_loss = 0
        print('Start epoch {}'.format(epoch))
        for i_batch, sample_batched in enumerate(train_loader):
            # https://pytorch.org/docs/stable/notes/cuda.html
            inputs = sample_batched['image'].cuda()
            labels = sample_batched['coor'].cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if i_batch % 1 == 0:    # every 20 mini-batches...

                print('Train batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.30f}'.format(
                        i_batch,
                        i_batch * len(inputs),
                        len(train_loader.dataset), 100. * i_batch / len(train_loader),
                        loss.item())) #/ len(inputs)))
                writer.add_scalar("training_loss", \
                        loss.item(), #/ len(inputs), \
                        i_batch  + epoch * math.ceil(len(train_loader) / batch_size) \
                        )

        print('epoch [{}/{}], training loss:{:.30f}'.format(epoch+1, num_epochs, \
                                         train_loss / len(train_loader)))

        model.eval()
        with torch.no_grad():
            total_acc_x = 0
            total_acc_y = 0
            for i, batch in enumerate(valid_loader):
                inputs = batch['image'].float().cuda()
                labels = batch['coor'].float().cuda()
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels)

                outputs = outputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                acc_x, acc_y = accuracy(outputs, labels)
                total_acc_x += acc_x
                total_acc_y += acc_y

            valid_loss = valid_loss / len(valid_loader)
            print('Valid loss {}'.format(valid_loss))

            writer.add_scalar("Valid_loss", \
                    valid_loss / len(valid_loader), \
                    epoch \
                    )

            print("=" * 30)
            print("total acc_x = {:.10f}".format(total_acc_x/len(valid_loader)))
            # set_trace()
            print("total acc_y = {:.10f}".format(total_acc_y/len(valid_loader)))
            print("=" * 30)

        torch.save(model.state_dict(), saved_weight_dir.format(str(epoch)))
        print('model saved to ' + saved_weight_dir.format(str(epoch)))

train()
