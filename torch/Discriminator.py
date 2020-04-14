import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from pdb import set_trace

C = 64


def weights_init_kaiming(m):
    classname = m.__class__name.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.uniform_(0.0, 1.0)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.enc1 = nn.Sequential(
                nn.Conv2d(2, C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(C, C, 3, padding=1),
                # nn.BatchNorm2d(C),
                # nn.InstanceNorm2d(C, affine=False),
                nn.ReLU(),
                )
        self.enc2 = nn.Sequential(
                nn.Conv2d(C, 2*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(2*C, 2*C, 3, padding=1),
                # nn.BatchNorm2d(2*C),
                # nn.InstanceNorm2d(2*C, affine=False),
                nn.ReLU(),
                )
        self.enc3 = nn.Sequential(
                nn.Conv2d(2*C, 4*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 3, padding=1),
                # nn.BatchNorm2d(4*C),
                # nn.InstanceNorm2d(4*C, affine=False),
                nn.ReLU(),
                )
        self.enc4 = nn.Sequential(
                nn.Conv2d(4*C, 8*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                # nn.InstanceNorm2d(8*C, affine=False),
                nn.ReLU(),
                )
        self.enc5 = nn.Sequential(
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                # nn.InstanceNorm2d(8*C, affine=False),
                nn.ReLU(),
                )

        self.linear1 = nn.Linear(8*C*7*7, 1000)
        kaiming_normal_(self.linear1.weight)
        self.linear2= nn.Linear(1000, 1)
        kaiming_normal_(self.linear2.weight)
        self.relu = nn.ReLU()

    def forward(self, inp):
        out1 = self.enc1(inp)
        out2 = self.enc2(F.max_pool2d(out1, 2))
        out3 = self.enc3(F.max_pool2d(out2, 2))
        out4 = self.enc4(F.max_pool2d(out3, 2))
        out5 = self.enc5(F.max_pool2d(out4, 2))
        x = F.max_pool2d(out5, 2)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.linear1(x)
        x = self.linear2(self.relu(x))
        x = torch.sigmoid(x)
        # x = nn.Softmax(1)(x)


        return x


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    # D = Discriminator().cuda()
    # import torch
    # input = torch.randn(3,2,224,224).cuda()
    # print(input)
    # print(input.shape)
    # output = D(input)
    # set_trace()

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # Create the Discriminator
    netD = Discriminator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)
