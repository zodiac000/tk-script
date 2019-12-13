import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from collections import OrderedDict
from pdb import set_trace

C = 64

class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.enc1 = nn.Sequential(
                nn.Conv2d(1, C, 3, padding=1),
                # nn.BatchNorm2d(C),
                nn.ReLU(),
                nn.Conv2d(C, C, 3, padding=1),
                # nn.BatchNorm2d(C),
                nn.ReLU(),
                )
        self.enc2 = nn.Sequential(
                nn.Conv2d(C, 2*C, 3, padding=1),
                # nn.BatchNorm2d(2*C),
                nn.ReLU(),
                nn.Conv2d(2*C, 2*C, 3, padding=1),
                # nn.BatchNorm2d(2*C),
                nn.ReLU(),
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
                )
        # self.enc6 = nn.Sequential(
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # )
        # self.enc7 = nn.Sequential(
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # )
        # self.enc8 = nn.Sequential(
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # nn.Conv2d(8*C, 8*C, 3, padding=1),
                # # nn.BatchNorm2d(8*C),
                # nn.ReLU(),
                # )
        self.dec5 = nn.ConvTranspose2d(8*C, 8*C, 2, stride=2) #512
        self.dec4 = nn.ConvTranspose2d(8*C, 4*C, 2, stride=2) #256
        self.dec3 = nn.ConvTranspose2d(4*C, 2*C, 2, stride=2) #128
        self.dec2 = nn.ConvTranspose2d(2*C, 1*C, 2, stride=2) #64
        self.dec1 = nn.ConvTranspose2d(1*C, 1, 2, stride=2) #512

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
        # self.conv_0 = nn.Conv2d(in_channels=1, out_channels=C, kernel_size=3, stride=1, padding=1, bias=True)
        # self.features = nn.Sequential(OrderedDict([('Block_0', Block(C))]))
        # self.num_of_blocks = 18
        # for i in range(1, self.num_of_blocks):
            # self.features.add_module('Block_%d' % i, Block(C))

        # self.fc1_1 = nn.Linear(7*7*8*C, 4096)
        # self.fc1_2 = nn.Linear(4096, 4096)
        # self.fc1_3 = nn.Linear(4096, 2)


        # self.fc2_1 = nn.Linear(7*7*8*C, 4096)
        # self.fc2_2 = nn.Linear(4096, 4096)
        # self.fc2_3 = nn.Linear(4096, 1)
    def forward(self, inp):
        # conv1 = F.max_pool2d(self.enc1(inp) + self.shortcut1(inp), 2)  #64*112*112
        # conv2 = F.max_pool2d(self.enc2(conv1) + self.shortcut2(conv1), 2)  #128*56*56
        # conv3 = F.max_pool2d(self.enc3(conv2) + self.shortcut3(conv2), 2)  #256*28*28
        # conv4 = F.max_pool2d(self.enc4(conv3) + self.shortcut4(conv3), 2)  #512*14*14
        # conv5 = F.max_pool2d(self.enc5(conv4) + self.shortcut5(conv4), 2)  #512*7*7
        # conv6 = self.enc6(conv5) + self.shortcut6(conv5)  #512*7*7
        # conv7 = self.enc7(conv6) + self.shortcut7(conv6)  #512*7*7
        # conv8 = self.enc8(conv7) + self.shortcut8(conv7)  #512*7*7
        
        # x = self.conv_0(inp)

        # for index, f in enumerate(self.features):
            # x = f(x)
        # x = F.max_pool2d(x, 2)
        # conv2 = F.max_pool2d(self.enc2(x), 2)  #128*56*56
        # conv3 = F.max_pool2d(self.enc3(conv2), 2)  #256*28*28
        # conv4 = F.max_pool2d(self.enc4(conv3), 2)  #512*14*14
        # conv5 = F.max_pool2d(self.enc5(conv4), 2)  #512*7*7

        conv1 = F.max_pool2d(self.enc1(inp), 2)  #64*112*112
        conv2 = F.max_pool2d(self.enc2(conv1), 2)  #128*56*56
        conv3 = F.max_pool2d(self.enc3(conv2), 2)  #256*28*28
        conv4 = F.max_pool2d(self.enc4(conv3), 2)  #512*14*14
        conv5 = F.max_pool2d(self.enc5(conv4), 2)  #512*7*7
        dfc1 = self.dec5(conv5)
        dfc2 = self.dec4(F.relu(dfc1+conv4))
        dfc3 = self.dec3(F.relu(dfc2+conv3))
        dfc4 = self.dec2(F.relu(dfc3+conv2))
        out = self.dec1(F.relu(dfc4+conv1))
        out = nn.Softmax(2)(out.view(-1, 1, 224*224)).view(-1, 1,224,224)

        # flatten1 = torch.flatten(conv5, start_dim=1)
        # fc1_1 = F.relu(self.fc1_1(flatten1))
        # fc1_2 = F.relu(self.fc1_2(fc1_1))
        # dx_dy = self.fc1_3(fc1_2)
        
        # flatten2 = torch.flatten(conv5, start_dim=1)
        # fc2_1 = F.relu(self.fc2_1(flatten2))
        # fc2_2 = F.relu(self.fc2_2(fc2_1))
        # out3 = nn.Sigmoid()(self.fc2_3(fc2_2))

        return out
        # return out, dx_dy

# class Block(nn.Module):
    # def __init__(self, C):
        # super(Block, self).__init__()
        # self.conv_3_1 = nn.Conv2d(in_channels=C, out_channels=2*C, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv_5_1 = nn.Conv2d(in_channels=C, out_channels=2*C, kernel_size=5, stride=1, padding=2, bias=True)
        # # self.conv_3_2 = nn.Conv2d(in_channels=4*C, out_channels=4*C, kernel_size=3, stride=1, padding=1, bias=True)
        # # self.conv_5_2 = nn.Conv2d(in_channels=4*C, out_channels=4*C, kernel_size=5, stride=1, padding=2, bias=True)
        # # self.confusion = nn.Conv2d(in_channels=8*C, out_channels=C, kernel_size=1, stride=1, padding=0, bias=False)

        # self.conv_3_2 = nn.Conv2d(in_channels=C, out_channels=2*C, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv_5_2 = nn.Conv2d(in_channels=C, out_channels=2*C, kernel_size=5, stride=1, padding=2, bias=True)
        # self.confusion_1 = nn.Conv2d(in_channels=4*C, out_channels=C, kernel_size=1, stride=1, padding=0, bias=False)
        # self.confusion_2 = nn.Conv2d(in_channels=4*C, out_channels=C, kernel_size=1, stride=1, padding=0, bias=False)

        # self.relu = nn.ReLU(inplace=True)

    # # def forward(self, x, pool=True):
        # # identity_data = x
        # # output_3_1 = self.relu(self.conv_3_1(x))
        # # output_5_1 = self.relu(self.conv_5_1(x))
        # # input_2 = torch.cat([output_3_1, output_5_1], 1)


        # # output_3_2 = self.relu(self.conv_3_2(input_2))
        # # output_5_2 = self.relu(self.conv_5_2(input_2))
        # # output = torch.cat([output_3_2, output_5_2], 1)

        # # output = self.confusion(output)
        # # output = torch.add(output, identity_data)
        # # return output

    # def forward(self, x, pool=True):
        # identity_data = x
        # output_3_1 = self.relu(self.conv_3_1(x))
        # output_5_1 = self.relu(self.conv_5_1(x))
        # input_2 = torch.cat([output_3_1, output_5_1], 1)

        # input_2 = self.confusion_1(input_2)

        # output_3_2 = self.relu(self.conv_3_2(input_2))
        # output_5_2 = self.relu(self.conv_5_2(input_2))
        # output = torch.cat([output_3_2, output_5_2], 1)

        # output = self.confusion_2(output)
        # output = torch.add(output, identity_data)
        # return output

if __name__ == "__main__":
    student = Student().cuda()
    input = torch.randn(2,1,224,224).cuda()
    print(input)
    print(input.shape)
    output = student(input)
    set_trace()

