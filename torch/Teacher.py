import torch
import torch.nn as nn
import torch.nn.functional as F

C = 64

class Teacher(nn.Module):
    def __init__(self):
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
        self.linear = nn.Linear(7*7*8*C, 1)

    def forward(self, inp):
        out1 = self.enc1(inp)
        out2 = self.enc2(F.maxpool2d(2, out1))
        out3 = self.enc3(F.maxpool2d(2, out2))
        out4 = self.enc4(F.maxpool2d(2, out3))
        out5 = self.enc5(F.maxpool2d(2, out4))
        x = self.linear(F.maxpool2d(2, out5))
        x = torch.sigmoid(x)





