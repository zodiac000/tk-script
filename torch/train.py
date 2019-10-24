from Student import Student
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import WeldingDataset
from pdb import set_trace
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math






########################################    Transformed Dataset


transformed_dataset = WeldingDataset(csv_file='./saved_dict.csv', root_dir='./')
                                    # transform=composed)

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['coor'].size())

    if i == 3:
        break

###########################################    Dataloader 

# dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, \
                        # num_workers=4)

num_epochs = 50
random_seed = 88
split = 1000
batch_size = 32
shuffle_dataset = True
indices = list(range(len(transformed_dataset)))

train_indices, val_indices = indices[:-split], indices[-split:]

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(train_indices)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(transformed_dataset, batch_size=batch_size, \
                        num_workers=6, sampler=train_sampler)
valid_loader = DataLoader(transformed_dataset, batch_size=batch_size, \
                        num_workers=6, sampler=valid_sampler)

writer = SummaryWriter('runs/experiment_1')

# for i_batch, sample_batched in enumerate(valid_loader):
    # print(i_batch, sample_batched['image'].size(),
          # sample_batched['coor'].size())

    # # observe 4th batch and stop.
    # if i_batch == 3:
        # plt.figure()
        # show_coor_batch(sample_batched)
        # plt.axis('off')
        # plt.ioff()
        # plt.show()
        # break





# get some random training images
# dataiter = iter(train_loader)
# batch = dataiter.next()
# images = batch['image']
# labels = batch['coor']

# create grid of images
# img_grid = torchvision.utils.make_grid(images)

# show images
# matplotlib_imshow(img_grid, one_channel=False)

# write to tensorboard
# writer.add_image('four_images', img_grid)

model = Student().cuda()
criterion = nn.MSELoss().cuda()
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                            # weight_decay=1e-5) # weight_decay

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    valid_loss = 0
    print('start epoch {}'.format(epoch))
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
        if i_batch % 20 == 0:    # every 20 mini-batches...

            # ...log the running loss
            # writer.add_scalar('training loss',
                            # train_loss / 10,
                            # epoch * len(train_loader) + i_batch)

            # print(train_loss / 100)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.30f}'.format(
                epoch,
                i_batch * len(inputs),
                len(train_loader.dataset), 100. * i_batch / len(train_loader),
                loss.item() / len(inputs)))
            writer.add_scalar("training_loss", \
                    loss.item() / len(inputs), \
                    i_batch  + epoch * math.ceil(len(train_loader) / batch_size) \
                    )


            # train_loss = 0
            # print('batch [{}], loss:{:.4f}'.format(i_batch * 10, loss.data()))
######################################################################
    # model.eval()
    # for i, batch in enumerate(valid_loader):
        # inputs = batch['image'].float().cuda()
        # labels = batch['coor'].float().cuda()
        # outputs = model(inputs)
        # valid_loss += criterion(outputs, labels)
    # valid_loss = valid_loss / len(valid_loader)
    # print('valid loss {}'.format(valid_loss))
    # valid_loss = 0
######################################################################

    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, \
                                     train_loss / len(train_loader.dataset)))
print('Finished Training')

torch.save(model.state_dict(), './vae.pth')
