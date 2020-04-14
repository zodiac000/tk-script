from Generator import Generator
from Discriminator import Discriminator
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import WeldingDatasetToTensor 
from pdb import set_trace
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
from random import uniform
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('tkagg')

ntrain = 200
nWorkers = 0
num_epochs = 2000
batch_size = 10
test_batch_size = 25
lr_D = 1e-4
lr_G = 1e-4
beta1 = 0.5

data_root = "all_images/"
csv_root = "csv/"
tb_root = "runs/"

train_csv = csv_root + 'pass_valid_' + str(ntrain) + '.csv'
print(train_csv)

# weight_to_load_student = './check_points/saved_weights_' + str(ntrain) + '.pth'


writer_gan_dir = tb_root + 'cgan_' + str(ntrain)

netG = Generator().cuda()
# netG.load_state_dict(torch.load(weight_to_load_student))

netD = Discriminator().cuda()
# netD.load_state_dict(torch.load(weight_to_load_teacher))

train_dataset = WeldingDatasetToTensor(data_root, csv_file=train_csv, root_dir='./')
train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                                num_workers=nWorkers, shuffle=True)

test_dataset = WeldingDatasetToTensor(data_root, csv_file=train_csv, root_dir='./')
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, \
                                num_workers=nWorkers, shuffle=True)

test_iterator = iter(test_loader)
test_samples = next(test_iterator)
test_hmaps = test_samples['hmap']
images = test_samples['image']

# vutils.save_image(images[0], 'images_G/real.png')
# set_trace()

fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (test_batch_size, 1, 224, 224)))
# vutils.save_image(fixed_noise[0], "images_G/noise.png")
# vutils.save_image(test_hmaps[0], "images_G/hmap.png")
cat_fixed_noise = torch.cat((fixed_noise, test_hmaps), 1)


writer_gan = SummaryWriter(writer_gan_dir)

# criterion = nn.MSELoss().cuda()
criterion = nn.BCELoss().cuda()
# criterion = nn.CrossEntropyLoss()


G_solver = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))
D_solver = torch.optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
# G_solver = torch.optim.RMSprop(netG.parameters(), lr=lr_G)
# D_solver = torch.optim.RMSprop(netD.parameters(), lr=lr_D)
# G_solver = torch.optim.Adadelta(netG.parameters(), lr=lr_G)
# D_solver = torch.optim.Adadelta(netD.parameters(), lr=lr_D)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator




# Plot some training images
# real_batch = next(iter(train_loader))
# plt.figure(figsize=(4,4))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(device)[:16], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()

def train_gan():
    correct_rate = 0
    nz = 100

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    img_list = []

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        print('Start epoch {} training on netD'.format(epoch))
        for i, batch_data in enumerate(train_loader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # netG.eval()
            # netD.train()
            netD.zero_grad()

            real_images = batch_data['image'].cuda()
            hmaps = batch_data['hmap'].cuda()


            b_size = real_images.size(0)
            noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (b_size, 1, 224, 224)))
            real_input_batch = None
            
            G_fake_inputs = torch.cat((noise, hmaps), 1)
            G_fake_outputs = netG.cuda()(G_fake_inputs)
            
            D_fake_inputs = torch.cat((G_fake_outputs, hmaps), 1)
            D_real_inputs = torch.cat((real_images, hmaps), 1)

            D_fake_outputs = netD(D_fake_inputs.detach())
            D_real_outputs = netD(D_real_inputs)

            D_x = D_real_outputs.view(-1).mean().item()
            D_G_z1 = D_fake_outputs.view(-1).mean().item()
            
            labels_real = torch.full((b_size, 1), real_label, device=device)
            D_loss_real = criterion(D_real_outputs, labels_real)
            labels_fake = torch.full((b_size, 1), fake_label, device=device)
            D_loss_fake = criterion(D_fake_outputs, labels_fake)
            D_loss = D_loss_real + D_loss_fake

            D_loss.backward()
            D_solver.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            # netG.train()
            # netD.eval()
            netG.zero_grad()
            # labels.fill_(real_label)

            labels_real = torch.full((b_size, 1), real_label, device=device)

            D_fake_outputs = netD(D_fake_inputs)
            G_loss = criterion(D_fake_outputs, labels_real)

            D_G_z2 = D_fake_outputs.view(-1).mean().item()
            G_loss.backward()
            G_solver.step()

            # total += b_size * 2
            # correct += (D_real_outputs>=0.5).sum().item() + (D_fake_outputs<0.5).sum().item()

            # Output training stats
            if epoch % 2 == 0:
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      # % (epoch, num_epochs, i, len(train_loader),
                      % (epoch, num_epochs, 
                         D_loss.item(), G_loss.item(), D_x, D_G_z1, D_G_z2))

                writer_gan.add_scalar("cgan_G_loss", \
                        G_loss.item(), #/ len(real_images), \
                        epoch * len(train_loader) + i \
                        )

                writer_gan.add_scalar("cgan_D_loss", \
                        D_loss.item(), #/ len(real_images), \
                        epoch * len(train_loader) + i \
                        )


                # if batch_correct_rate > correct_rate:
                    # correct_rate = batch_correct_rate
                dir_weight = './check_points/cgan_G.pth'
                torch.save(netG.state_dict(), dir_weight)
                print('Generator weights saved to ' + dir_weight)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (epoch % 10 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake = netG.cpu()(cat_fixed_noise).detach()
                # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    for idx, f in enumerate(fake):
                        vutils.save_image(f, 'images_G/' + str(epoch) + '_' + str(idx) + '.png')






if __name__ == "__main__":
    train_gan()

