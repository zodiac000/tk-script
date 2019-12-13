import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from collections import OrderedDict
from pdb import set_trace

C = 64

class Student_cls(nn.Module):
    def __init__(self):
        super(Student_cls, self).__init__()
        self.enc1 = nn.Sequential(
                nn.Conv2d(1, C, 3, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(),
                nn.Conv2d(C, C, 3, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(),
                )
        self.enc2 = nn.Sequential(
                nn.Conv2d(C, 2*C, 3, padding=1),
                nn.BatchNorm2d(2*C),
                nn.ReLU(),
                nn.Conv2d(2*C, 2*C, 3, padding=1),
                nn.BatchNorm2d(2*C),
                nn.ReLU(),
                )
        self.enc3 = nn.Sequential(
                nn.Conv2d(2*C, 4*C, 3, padding=1),
                nn.BatchNorm2d(4*C),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 3, padding=1),
                nn.BatchNorm2d(4*C),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 3, padding=1),
                nn.BatchNorm2d(4*C),
                nn.ReLU(),
                )
        self.enc4 = nn.Sequential(
                nn.Conv2d(4*C, 8*C, 3, padding=1),
                nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.BatchNorm2d(8*C),
                nn.ReLU(),
                )
        self.enc5 = nn.Sequential(
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                nn.BatchNorm2d(8*C),
                nn.ReLU(),
                )

        self.fc1 = nn.Linear(7*7*8*C, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 1)

    def forward(self, inp):
        conv1 = F.max_pool2d(self.enc1(inp), 2)  #64*112*112
        conv2 = F.max_pool2d(self.enc2(conv1), 2)  #128*56*56
        conv3 = F.max_pool2d(self.enc3(conv2), 2)  #256*28*28
        conv4 = F.max_pool2d(self.enc4(conv3), 2)  #512*14*14
        conv5 = F.max_pool2d(self.enc5(conv4), 2)  #512*7*7

        flatten = torch.flatten(conv5, start_dim=1)
        fc1 = F.relu(self.fc1(flatten))
        fc2 = F.relu(self.fc2(fc1))
        fc3 = F.relu(self.fc3(fc2))
        out = nn.Sigmoid()(self.fc4(fc3))
        
        return out

if __name__ == "__main__":
    student = Student().cuda()
    input = torch.randn(2,1,224,224).cuda()
    print(input)
    print(input.shape)
    output = student(input)
    set_trace()

