import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from collections import OrderedDict
from pdb import set_trace

C = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.enc1 = nn.Sequential(
                nn.Conv2d(2, C, 3, padding=1),
                # nn.BatchNorm2d(C),
                nn.ReLU(),
                nn.Conv2d(C, C, 3, padding=1),
                # nn.BatchNorm2d(C),
                nn.ReLU(),
                # nn.Conv2d(C, C, 3, padding=1),
                # # nn.BatchNorm2d(C),
                # nn.ReLU(),
                )
        self.enc2 = nn.Sequential(
                nn.Conv2d(C, 2*C, 3, padding=1),
                # nn.BatchNorm2d(2*C),
                nn.ReLU(),
                nn.Conv2d(2*C, 2*C, 3, padding=1),
                # nn.BatchNorm2d(2*C),
                nn.ReLU(),
                # nn.Conv2d(2*C, 2*C, 3, padding=1),
                # # nn.BatchNorm2d(2*C),
                # nn.ReLU(),
                )
        self.enc3 = nn.Sequential(
                nn.Conv2d(2*C, 4*C, 3, padding=1),
                # nn.BatchNorm2d(4*C),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 3, padding=1),
                # nn.BatchNorm2d(4*C),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 3, padding=1),
                # nn.BatchNorm2d(4*C),
                nn.ReLU(),
                # nn.Conv2d(4*C, 4*C, 3, padding=1),
                # # nn.BatchNorm2d(4*C),
                # nn.ReLU(),
                )
        self.enc4 = nn.Sequential(
                nn.Conv2d(4*C, 8*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                )
        self.enc5 = nn.Sequential(
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                )
        self.dec5 = nn.ConvTranspose2d(8*C, 8*C, 2, stride=2) #512
        self.dec4 = nn.ConvTranspose2d(8*C, 4*C, 2, stride=2) #256
        self.dec3 = nn.ConvTranspose2d(4*C, 2*C, 2, stride=2) #128
        self.dec2 = nn.ConvTranspose2d(2*C, 1*C, 2, stride=2) #64
        self.dec1 = nn.ConvTranspose2d(1*C, 1, 2, stride=2) #512

    def forward(self, inp):
        conv1 = F.max_pool2d(self.enc1(inp), 2)  #64*112*112  ----------------------|
        conv2 = F.max_pool2d(self.enc2(conv1), 2)  #128*56*56 ----------------|     |
        conv3 = F.max_pool2d(self.enc3(conv2), 2)  #256*28*28 -----------     |     |
        conv4 = F.max_pool2d(self.enc4(conv3), 2)  #512*14*14 -------    |    |     |
        conv5 = F.max_pool2d(self.enc5(conv4), 2)  #512*7*7          |   |    |     |
#                                                                    |   |    |     |
#                                                                    |   |    |     |
        dfc1 = self.dec5(conv5)               #                      |   |    |     |
        dfc2 = self.dec4(F.relu(dfc1+conv4))  #     -----------------|   |    |     |
        dfc3 = self.dec3(F.relu(dfc2+conv3))  #   -----------------------|    |     |
        dfc4 = self.dec2(F.relu(dfc3+conv2))  #   ----------------------------|     |  
        out = self.dec1(F.relu(dfc4+conv1))   #  -----------------------------------|
        # out = nn.Softmax(2)(out.view(-1, 1, 224*224)).view(-1, 1,224,224)


        return out

if __name__ == "__main__":
    generator = Generator().cuda()
    input = torch.randn(2,1,224,224).cuda()
    print(input)
    print(input.shape)
    output = generator(input)
    set_trace()

