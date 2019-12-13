from Student_reg import Student
# from Student2 import Student2
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

# training_number = 50
# training_number = 4265
training_number = 6500
# training_number = 6415

train_dataset = WeldingDatasetToTensor(csv_file='./csv/pass_adbc_' + str(training_number) + '.csv', \
                                        root_dir='./')
# val_dataset = WeldingDatasetToTensor(csv_file='./csv/pass_valid_1000.csv', root_dir='./')
val_dataset = WeldingDatasetToTensor(csv_file='./csv/pass_adbc_1000.csv', root_dir='./')
# val_dataset = WeldingDatasetToTensor(csv_file='./csv/tail_100.csv', root_dir='./')
saved_weight_dir = './check_points/saved_weights_ad_' + str(training_number) + '.pth'
tensorboard_file = 'runs/cas_ad_' + str(training_number)

# saved_weights = './check_points/saved_weights_200.pth'
for i in range(len(train_dataset)):
    sample = train_dataset[i]
    print(i, sample['image'].size(), sample['hmap'].size())
    if i == 3:
        break

num_epochs = 30000
batch_size = 30
lr = 1e-3

train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                        num_workers=4, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, \
                        num_workers=4)

writer = SummaryWriter(tensorboard_file)


model = Student().cuda()
# model = Student2().cuda()
mse = nn.MSELoss().cuda()
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def train():
    train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                        num_workers=4, shuffle=True)
    # model.load_state_dict(torch.load(saved_weights))
    max_total_acc_x = 0
    max_euclidean_distance = 99999
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
        labels = sample_batched['hmap'].cuda()
        coors_bc = sample_batched['coor_bc'].cpu().detach().numpy()
        # class_real = sample_batched['class_real'].cuda()

        img_names = sample_batched['img_name']
        origin_imgs = sample_batched['origin_img']


        optimizer.zero_grad()
        outputs = model(inputs)

        loss = mse(outputs, labels)

        # loss_bce = nn.functional.binary_cross_entropy(class_pred, class_real)
        # loss = (1-class_real) * loss_bce + class_real * (l * loss_bce + (1-l)*loss_mse)
        # torch.mean(loss).backward()
        loss.backward()
        optimizer.step()
        
############################  896 block  ############
        empty_batch = True
        for index, out in enumerate(outputs.cpu().detach().numpy()):
            w_center, h_center = heatmap_to_coor(out.squeeze())
            cropped_image, cropped_hmap = crop(origin_imgs[index], w_center, h_center, coors_bc[index], 224*4)
            
            cropped_image = ToTensor()(Resize((224,224))(Image.fromarray(cropped_image.numpy()))).unsqueeze(0)
            cropped_hmap = cropped_hmap.unsqueeze(dim=0).unsqueeze(0)


            if empty_batch:
                cropped_image_batch = cropped_image
                cropped_hmap_batch = cropped_hmap
                empty_batch = False
            else:
                cropped_image_batch = torch.cat([cropped_image_batch, cropped_image])
                cropped_hmap_batch = torch.cat([cropped_hmap_batch, cropped_hmap])

        optimizer.zero_grad()
        outputs = model(cropped_image_batch.cuda())
        loss = mse(outputs, cropped_hmap_batch.cuda())

        torch.mean(loss).backward()
        # loss.backward()
        optimizer.step()
############################  448 block  ############
        empty_batch = True
        for index, out in enumerate(outputs.cpu().detach().numpy()):
            w_center, h_center = heatmap_to_coor(out.squeeze())
            cropped_image, cropped_hmap = crop(origin_imgs[index], w_center, h_center, coors_bc[index], 224*2)
            
            cropped_image = ToTensor()(Resize((224,224))(Image.fromarray(cropped_image.numpy()))).unsqueeze(0)
            cropped_hmap = cropped_hmap.unsqueeze(dim=0).unsqueeze(0)


            if empty_batch:
                cropped_image_batch = cropped_image
                cropped_hmap_batch = cropped_hmap
                empty_batch = False
            else:
                cropped_image_batch = torch.cat([cropped_image_batch, cropped_image])
                cropped_hmap_batch = torch.cat([cropped_hmap_batch, cropped_hmap])

        optimizer.zero_grad()
        outputs = model(cropped_image_batch.cuda())
        loss = mse(outputs, cropped_hmap_batch.cuda())
        torch.mean(loss).backward()
        # loss.backward()
        optimizer.step()
############################  224 block  ############
        empty_batch = True
        for index, out in enumerate(outputs.cpu().detach().numpy()):
            w_center, h_center = heatmap_to_coor(out.squeeze())
            cropped_image, cropped_hmap = crop(origin_imgs[index], w_center, h_center, coors_bc[index], 224)
            cropped_image = ToTensor()(cropped_image.unsqueeze(dim=-1).numpy()).unsqueeze(0)
            cropped_hmap = cropped_hmap.unsqueeze(dim=0).unsqueeze(0)


            if empty_batch:
                cropped_image_batch = cropped_image
                cropped_hmap_batch = cropped_hmap
                empty_batch = False
            else:
                cropped_image_batch = torch.cat([cropped_image_batch, cropped_image])
                cropped_hmap_batch = torch.cat([cropped_hmap_batch, cropped_hmap])

        optimizer.zero_grad()
        outputs = model(cropped_image_batch.cuda())
        loss = mse(outputs, cropped_hmap_batch.cuda())
        
        # torch.mean(loss).backward()
        loss.backward()
        optimizer.step()

############################  112 block  ############
        empty_batch = True
        for index, out in enumerate(outputs.cpu().detach().numpy()):
            w_center, h_center = heatmap_to_coor(out.squeeze())
            cropped_image, cropped_hmap = crop(origin_imgs[index], w_center, h_center, coors_bc[index], int(224/2))
            
            cropped_image = ToTensor()(Resize((224,224))(Image.fromarray(cropped_image.numpy()))).unsqueeze(0)
            cropped_hmap = cropped_hmap.unsqueeze(dim=0).unsqueeze(0)

            if empty_batch:
                cropped_image_batch = cropped_image
                cropped_hmap_batch = cropped_hmap
                empty_batch = False
            else:
                cropped_image_batch = torch.cat([cropped_image_batch, cropped_image])
                cropped_hmap_batch = torch.cat([cropped_hmap_batch, cropped_hmap])

        optimizer.zero_grad()
        outputs = model(cropped_image_batch.cuda())
        loss = mse(outputs, cropped_hmap_batch.cuda())
        torch.mean(loss).backward()
        optimizer.step()

############################  56 block  ############
        empty_batch = True
        for index, out in enumerate(outputs.cpu().detach().numpy()):
            w_center, h_center = heatmap_to_coor(out.squeeze())
            cropped_image, cropped_hmap = crop(origin_imgs[index], w_center, h_center, coors_bc[index], int(224/4))
            
            cropped_image = ToTensor()(Resize((224,224))(Image.fromarray(cropped_image.numpy()))).unsqueeze(0)
            cropped_hmap = cropped_hmap.unsqueeze(dim=0).unsqueeze(0)

            if empty_batch:
                cropped_image_batch = cropped_image
                cropped_hmap_batch = cropped_hmap
                empty_batch = False
            else:
                cropped_image_batch = torch.cat([cropped_image_batch, cropped_image])
                cropped_hmap_batch = torch.cat([cropped_hmap_batch, cropped_hmap])

        optimizer.zero_grad()
        outputs = model(cropped_image_batch.cuda())
        loss = mse(outputs, cropped_hmap_batch.cuda())
        torch.mean(loss).backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:    # every 20 mini-batches...
            e_distance = 0
            for index, out in enumerate(outputs):
                x, y = heatmap_to_coor(out.reshape(224, 224).cpu().detach().numpy())
                e_distance += ((int(x/224*1280)-coors_bc[index][0])**2 + \
                                (int(y/224*1024)-coors_bc[index][1])**2)**0.5

            print('Train epoch: {}\tLoss: {:.30f}'.format(
                    epoch+1,
                    # i_batch * len(inputs),
                    # len(train_loader.dataset), 100. * i_batch / len(train_loader),
                    torch.mean(loss).item())) #/ len(inputs)))
            writer.add_scalar("cascade4_training_loss", \
                    torch.mean(loss).item(), #/ len(inputs), \
                    epoch  + epoch * math.ceil(len(train_loader) / batch_size) \
                    )
            writer.add_scalar("cascade4_training_Euclidean_Distance", \
                    e_distance, 
                    epoch  + epoch * math.ceil(len(train_loader) / batch_size) \
                    )


        if (epoch+1) % 50 == 0:    # every 20 mini-batches...
            # model.eval()
            with torch.no_grad():
                valid_loss = 0
                total_acc_x = 0
                total_acc_y = 0
                e_distance = 0
                for i, batch in enumerate(valid_loader):
                    inputs = batch['image'].float().cuda()
                    labels = batch['hmap'].float().cuda()
                    coors_bc = batch['coor_bc'].cpu().detach().numpy()

                    outputs = model(inputs)
                    loss = mse(outputs, labels)

                    outputs = outputs.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    
                    sum_acc_x, sum_acc_y, list_acc_x, list_acc_y = accuracy_sum(outputs, coors_bc)
                    total_acc_x += sum_acc_x
                    total_acc_y += sum_acc_y
                    # all_acc_x.extend(list_acc_x)
                    # all_acc_y.extend(list_acc_y)

                    for index, out in enumerate(outputs):
                        x, y = heatmap_to_coor(out.reshape(224, 224))
                        e_distance += ((int(x/224*1280)-coors_bc[index][0])**2 + \
                                        (int(y/224*1024)-coors_bc[index][1])**2)**0.5


                valid_loss = valid_loss / len(valid_loader)
                print('Valid loss {}'.format(valid_loss))

                writer.add_scalar("Valid_loss_adbc", valid_loss, epoch)
                writer.add_scalar("Valid_adbc_Euclidean_Distance", e_distance/len(valid_loader.dataset), epoch)

                print("=" * 30)
                print("total acc_x = {:.10f}".format(total_acc_x/len(valid_loader.dataset)))
                print("total acc_y = {:.10f}".format(total_acc_y/len(valid_loader.dataset)))
                print("Euclidean Distance: {}".format(e_distance/len(valid_loader.dataset)))
                print("=" * 30)
            
                # if total_acc_x > max_total_acc_x:
                    # max_total_acc_x = total_acc_x
                if e_distance/len(valid_loader.dataset) < max_euclidean_distance:
                    max_euclidean_distance = e_distance/len(valid_loader.dataset)
                    torch.save(model.state_dict(), saved_weight_dir)
                    print('model saved to ' + saved_weight_dir)

train()
