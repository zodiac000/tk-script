import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from collections import OrderedDict
from pdb import set_trace

C = 64

class Student2(nn.Module):
    def __init__(self):
        super(Student2, self).__init__()
        self.enc_3_1 = nn.Sequential(
                nn.Conv2d(1, C, 3, padding=1),
                # nn.BatchNorm2d(C),
                nn.ReLU(),
                nn.Conv2d(C, C, 3, padding=1),
                # nn.BatchNorm2d(C),
                nn.ReLU(),
                )
        self.enc_3_2 = nn.Sequential(
                nn.Conv2d(2*C, 2*C, 3, padding=1),
                # nn.BatchNorm2d(2*C),
                nn.ReLU(),
                nn.Conv2d(2*C, 2*C, 3, padding=1),
                # nn.BatchNorm2d(2*C),
                nn.ReLU(),
                )
        self.enc_3_3 = nn.Sequential(
                nn.Conv2d(4*C, 4*C, 3, padding=1),
                # nn.BatchNorm2d(4*C),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 3, padding=1),
                # nn.BatchNorm2d(4*C),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 3, padding=1),
                # nn.BatchNorm2d(4*C),
                nn.ReLU(),
                )
        self.enc_3_4 = nn.Sequential(
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                )
        self.enc_3_5 = nn.Sequential(
                nn.Conv2d(16*C, 16*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(16*C, 16*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(16*C, 16*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                )
        # self.enc_3_6 = nn.Sequential(
                # nn.Conv2d(16*C, 16*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # nn.Conv2d(16*C, 16*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # nn.Conv2d(16*C, 16*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # )
        self.enc6 = nn.Sequential(
                nn.Conv2d(32*C, 32*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(32*C, 32*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(32*C, 32*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                )
        self.enc7 = nn.Sequential(
                nn.Conv2d(32*C, 32*C, 3, padding=1),
                # nn.BatchNorm2d(32*C),
                nn.ReLU(),
                nn.Conv2d(32*C, 32*C, 3, padding=1),
                # nn.BatchNorm2d(32*C),
                nn.ReLU(),
                nn.Conv2d(32*C, 32*C, 3, padding=1),
                # nn.BatchNorm2d(32*C),
                nn.ReLU(),
                )
        self.enc8 = nn.Sequential(
                nn.Conv2d(32*C, 32*C, 3, padding=1),
                # nn.BatchNorm2d(32*C),
                nn.ReLU(),
                nn.Conv2d(32*C, 32*C, 3, padding=1),
                # nn.BatchNorm2d(32*C),
                nn.ReLU(),
                nn.Conv2d(32*C, 32*C, 3, padding=1),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                )
        # self.dec5 = nn.ConvTranspose2d(8*C, 8*C, 2, stride=2) #512
        # self.dec4 = nn.ConvTranspose2d(8*C, 4*C, 2, stride=2) #256
        # self.dec3 = nn.ConvTranspose2d(4*C, 2*C, 2, stride=2) #128
        # self.dec2 = nn.ConvTranspose2d(2*C, 1*C, 2, stride=2) #64
        # self.dec1 = nn.ConvTranspose2d(1*C, 1, 2, stride=2) #512

        self.dec5 = nn.ConvTranspose2d(32*C, 16*C, 2, stride=2) #512
        self.dec4 = nn.ConvTranspose2d(16*C, 8*C, 2, stride=2) #256
        self.dec3 = nn.ConvTranspose2d(8*C, 4*C, 2, stride=2) #128
        self.dec2 = nn.ConvTranspose2d(4*C, 2*C, 2, stride=2) #64
        self.dec1 = nn.ConvTranspose2d(2*C, C, 2, stride=2) #512
        self.dec0 = nn.ConvTranspose2d(C, 1, 1, stride=1) #512

        self.enc_5_1 = nn.Sequential(
                nn.Conv2d(1, C, 5, padding=2),
                # nn.BatchNorm2d(C),
                nn.ReLU(),
                nn.Conv2d(C, C, 5, padding=2),
                # nn.BatchNorm2d(C),
                nn.ReLU(),
                )
        self.enc_5_2 = nn.Sequential(
                nn.Conv2d(2*C, 2*C, 5, padding=2),
                # nn.BatchNorm2d(2*C),
                nn.ReLU(),
                nn.Conv2d(2*C, 2*C, 5, padding=2),
                # nn.BatchNorm2d(2*C),
                nn.ReLU(),
                )
        self.enc_5_3 = nn.Sequential(
                nn.Conv2d(4*C, 4*C, 5, padding=2),
                # nn.BatchNorm2d(4*C),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 5, padding=2),
                # nn.BatchNorm2d(4*C),
                nn.ReLU(),
                nn.Conv2d(4*C, 4*C, 5, padding=2),
                # nn.BatchNorm2d(4*C),
                nn.ReLU(),
                )
        self.enc_5_4 = nn.Sequential(
                nn.Conv2d(8*C, 8*C, 5, padding=2),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 5, padding=2),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                nn.Conv2d(8*C, 8*C, 5, padding=2),
                # nn.BatchNorm2d(8*C),
                nn.ReLU(),
                )
        self.enc_5_5 = nn.Sequential(
                nn.Conv2d(16*C, 16*C, 5, padding=2),
                # nn.BatchNorm2d(16*C),
                nn.ReLU(),
                nn.Conv2d(16*C, 16*C, 5, padding=2),
                # nn.BatchNorm2d(16*C),
                nn.ReLU(),
                nn.Conv2d(16*C, 16*C, 5, padding=2),
                # nn.BatchNorm2d(16*C),
                nn.ReLU(),
                )
        # self.enc_5_6 = nn.Sequential(
                # nn.Conv2d(16*C, 16*C, 5, padding=2),
                # # nn.BatchNorm2d(16*C),
                # nn.ReLU(),
                # nn.Conv2d(16*C, 16*C, 5, padding=2),
                # # nn.BatchNorm2d(16*C),
                # nn.ReLU(),
                # nn.Conv2d(16*C, 16*C, 5, padding=2),
                # # nn.BatchNorm2d(16*C),
                # nn.ReLU(),
                # )
        # self.shortcut1 = nn.Sequential(
                # nn.Conv2d(1, C, 1),
                # nn.BatchNorm2d(C),
                # )
        # self.shortcut2 = nn.Sequential(
                # nn.Conv2d(C, 2*C, 1),
                # nn.BatchNorm2d(2*C),
                # )
        # self.shortcut3 = nn.Sequential(
                # nn.Conv2d(2*C, 4*C, 1),
                # nn.BatchNorm2d(4*C),
                # )
        # self.shortcut4 = nn.Sequential(
                # nn.Conv2d(4*C, 8*C, 1),
                # nn.BatchNorm2d(8*C),
                # )
        # self.shortcut5 = nn.Sequential(
                # nn.Conv2d(8*C, 8*C, 1),
                # nn.BatchNorm2d(8*C),
                # )
        # self.shortcut6 = nn.Sequential(
                # nn.Conv2d(8*C, 8*C, 1),
                # nn.BatchNorm2d(8*C),
                # )
        # self.shortcut7 = nn.Sequential(
                # nn.Conv2d(8*C, 8*C, 1),
                # nn.BatchNorm2d(8*C),
                # )
        # self.shortcut8 = nn.Sequential(
                # nn.Conv2d(8*C, 8*C, 1),
                # nn.BatchNorm2d(8*C),
                # )
        self.conv_bottleneck = nn.Conv2d(in_channels=32*C, out_channels=8*C, kernel_size=1, stride=1, padding=0, bias=True)
        # self.conv_0 = nn.Conv2d(in_channels=1, out_channels=C, kernel_size=3, stride=1, padding=1, bias=True)
        # self.features = nn.Sequential(OrderedDict([('Block_0', Block(C))]))
        # self.num_of_blocks = 18
        # for i in range(1, self.num_of_blocks):
            # self.features.add_module('Block_%d' % i, Block(C))
    def forward(self, inp):
        # conv1 = F.max_pool2d(self.enc_3_1(inp) + self.shortcut1(inp), 2)  #64*112*112
        # conv2 = F.max_pool2d(self.enc_3_2(conv1) + self.shortcut2(conv1), 2)  #128*56*56
        # conv3 = F.max_pool2d(self.enc_3_3(conv2) + self.shortcut3(conv2), 2)  #256*28*28
        # conv4 = F.max_pool2d(self.enc_3_4(conv3) + self.shortcut4(conv3), 2)  #512*14*14
        # conv5 = F.max_pool2d(self.enc_3_5(conv4) + self.shortcut5(conv4), 2)  #512*7*7
        # conv6 = self.enc6(conv5) + self.shortcut6(conv5)  #512*7*7
        # conv7 = self.enc7(conv6) + self.shortcut7(conv6)  #512*7*7
        # conv8 = self.enc8(conv7) + self.shortcut8(conv7)  #512*7*7
        
        # x = self.conv_0(inp)

        # for index, f in enumerate(self.features):
            # x = f(x)
        # x = F.max_pool2d(x, 2)
        # conv2 = F.max_pool2d(self.enc_3_2(x), 2)  #128*56*56
        # conv3 = F.max_pool2d(self.enc_3_3(conv2), 2)  #256*28*28
        # conv4 = F.max_pool2d(self.enc_3_4(conv3), 2)  #512*14*14
        # conv5 = F.max_pool2d(self.enc_3_5(conv4), 2)  #512*7*7

        conv_3_1 = F.max_pool2d(self.enc_3_1(inp), 2)  #128*56*56
        conv_5_1 = F.max_pool2d(self.enc_5_1(inp), 2)  #128*56*56
        inp_2 = torch.cat([conv_3_1, conv_5_1], 1)
        conv_3_2 = F.max_pool2d(self.enc_3_2(inp_2), 2)  #128*56*56
        conv_5_2 = F.max_pool2d(self.enc_5_2(inp_2), 2)  #128*56*56
        inp_3 = torch.cat([conv_3_2, conv_5_2], 1)
        conv_3_3 = F.max_pool2d(self.enc_3_3(inp_3), 2)  #256*28*28
        conv_5_3 = F.max_pool2d(self.enc_5_3(inp_3), 2)  #256*28*28
        inp_4 = torch.cat([conv_3_3, conv_5_3], 1)
        conv_3_4 = F.max_pool2d(self.enc_3_4(inp_4), 2)  #512*14*14
        conv_5_4 = F.max_pool2d(self.enc_5_4(inp_4), 2)  #512*14*14
        inp_5 = torch.cat([conv_3_4, conv_5_4], 1)
        conv_3_5 = F.max_pool2d(self.enc_3_5(inp_5), 2)  #512*7*7
        conv_5_5 = F.max_pool2d(self.enc_5_5(inp_5), 2)  #512*7*7
        inp_6 = torch.cat([conv_3_5, conv_5_5], 1)

        # bottleneck = self.conv_bottleneck(inp_6)
        enc6 = self.enc6(inp_6)
        enc7 = self.enc6(enc6)
        enc8 = self.enc6(enc7)


        d5 = self.dec5(enc8)
        d4 = self.dec4(F.relu(d5 + inp_5))
        d3 = self.dec3(F.relu(d4 + inp_4))
        d2 = self.dec2(F.relu(d3 + inp_3))
        d1 = self.dec1(F.relu(d2 + inp_2))
        out = self.dec0(F.relu(d1))

        # dfc1 = self.dec5(bottleneck)
        # dfc2 = self.dec4(F.relu(dfc1))
        # dfc3 = self.dec3(F.relu(dfc2))
        # dfc4 = self.dec2(F.relu(dfc3))
        # out = self.dec1(F.relu(dfc4))
        out = nn.Softmax(2)(out.view(-1, 1, 224*224)).view(-1, 1,224,224)
        return out

class Block(nn.Module):
    def __init__(self, C):
        super(Block, self).__init__()
        self.conv_3_1 = nn.Conv2d(in_channels=C, out_channels=2*C, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=C, out_channels=2*C, kernel_size=5, stride=1, padding=2, bias=True)
        # self.conv_3_2 = nn.Conv2d(in_channels=4*C, out_channels=4*C, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv_5_2 = nn.Conv2d(in_channels=4*C, out_channels=4*C, kernel_size=5, stride=1, padding=2, bias=True)
        # self.confusion = nn.Conv2d(in_channels=8*C, out_channels=C, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_3_2 = nn.Conv2d(in_channels=C, out_channels=2*C, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_5_2 = nn.Conv2d(in_channels=C, out_channels=2*C, kernel_size=5, stride=1, padding=2, bias=True)
        self.confusion_1 = nn.Conv2d(in_channels=4*C, out_channels=C, kernel_size=1, stride=1, padding=0, bias=False)
        self.confusion_2 = nn.Conv2d(in_channels=4*C, out_channels=C, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)

    # def forward(self, x, pool=True):
        # identity_data = x
        # output_3_1 = self.relu(self.conv_3_1(x))
        # output_5_1 = self.relu(self.conv_5_1(x))
        # input_2 = torch.cat([output_3_1, output_5_1], 1)


        # output_3_2 = self.relu(self.conv_3_2(input_2))
        # output_5_2 = self.relu(self.conv_5_2(input_2))
        # output = torch.cat([output_3_2, output_5_2], 1)

        # output = self.confusion(output)
        # output = torch.add(output, identity_data)
        # return output

    def forward(self, x, pool=True):
        identity_data = x
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))
        input_2 = torch.cat([output_3_1, output_5_1], 1)

        input_2 = self.confusion_1(input_2)

        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        output = torch.cat([output_3_2, output_5_2], 1)

        output = self.confusion_2(output)
        output = torch.add(output, identity_data)
        return output

if __name__ == "__main__":
    student = Student2().cuda()
    input = torch.randn(2,1,224,224).cuda()
    print(input)
    print(input.shape)
    output = student(input)
    set_trace()

