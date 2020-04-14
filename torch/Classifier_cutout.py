import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from pdb import set_trace
from torchsummary import summary

C = 64


def weights_init_kaiming(m):
    classname = m.__class__name.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.uniform_(0.0, 1.0)



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.enc1 = nn.Sequential(
                nn.Conv2d(1, C, 3, padding=1),
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
        # self.enc6 = nn.Sequential(
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.ReLU(),
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.ReLU(),
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.ReLU(),
                # )
        # self.linear1 = nn.Linear(256*28*28, 1)
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

if __name__ == "__main__":
    classifier = Classifier().cuda()
    size = 224
    import torch
    noise = torch.randn(3,1,size,size).cuda()
    summary(classifier, (1,size,size))
    output = classifier(noise)

    set_trace()
