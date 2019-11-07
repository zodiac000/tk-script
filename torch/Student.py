import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from pdb import set_trace

C = 64
fc = 4096

class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.enc1 = nn.Sequential(
                nn.Conv2d(1, C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(C, C, 3, padding=1),
                nn.ReLU(),
                )
        self.enc2 = nn.Sequential(
                nn.Conv2d(C, 2*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(2*C, 2*C, 3, padding=1),
                nn.ReLU(),
                )
        self.enc3 = nn.Sequential(
                nn.Conv2d(2*C, 4*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 3, padding=1),
                nn.ReLU(),
                )
        self.enc4 = nn.Sequential(
                nn.Conv2d(4*C, 8*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.ReLU(),
                )
        self.enc5 = nn.Sequential(
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.ReLU(),
                )
        self.dec5 = nn.ConvTranspose2d(8*C, 8*C, 2, stride=2) #512
        self.dec4 = nn.ConvTranspose2d(8*C, 4*C, 2, stride=2) #256
        self.dec3 = nn.ConvTranspose2d(4*C, 2*C, 2, stride=2) #128
        self.dec2 = nn.ConvTranspose2d(2*C, 1*C, 2, stride=2) #64
        self.dec1 = nn.ConvTranspose2d(1*C, 1, 2, stride=2) #512

        # self.fc1 = nn.Linear(7*7*8*C, fc)
        # self.fc2 = nn.Linear(fc, fc)

    def forward(self, inp):
        conv1 = F.max_pool2d(self.enc1(inp), 2)  #64*112*112
        conv2 = F.max_pool2d(self.enc2(conv1), 2)  #128*56*56
        conv3 = F.max_pool2d(self.enc3(conv2), 2)  #256*28*28
        conv4 = F.max_pool2d(self.enc4(conv3), 2)  #512*14*14
        conv5 = F.max_pool2d(self.enc5(conv4), 2)  #512*7*7
        x = torch.flatten(conv5, start_dim=1)
        # x = conv5.reshape(-1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))

        dfc1 = self.dec5(conv5)
        # dfc1 = F.relu(dfc1)  #512*14*14
        dfc2 = self.dec4(F.relu(dfc1+conv4))
        # dfc2 = F.relu(dfc2)  #256*28*28
        dfc3 = self.dec3(F.relu(dfc2+conv3))
        # dfc3 = F.relu(dfc3)  #128*56*56
        dfc4 = self.dec2(F.relu(dfc3+conv2))
        # dfc4 = F.relu(dfc4)  #64*112*112
        out = self.dec1(F.relu(dfc4+conv1))
        # out = torch.sigmoid(out)  #1*224*224
        # set_trace()
        # out = nn.Softmax2d()(out)
        out = nn.Softmax(2)(out.view(-1, 1, 224*224)).view(-1, 1,224,224)
        return out


if __name__ == "__main__":
    student = Student().cuda()
    input = torch.randn(2,1,224,224).cuda()
    print(input)
    print(input.shape)
    output = student(input)
    set_trace()

