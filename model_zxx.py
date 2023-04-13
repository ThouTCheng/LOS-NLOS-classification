import torch.nn
import torch.nn as nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
    def __init__(self,input_size):
        super(FullyConnected, self).__init__()
        self.hidden1 = nn.Linear(input_size,64).double()
        self.hidden2 = nn.Linear(64,32).double()
    def forward(self, x):
        x = F.relu( self.hidden1(x))
        x = F.relu( self.hidden2(x))
        return x

class Model_ericssion_one(nn.Module):
    def __init__(self,num_clase,in_channel=[2,4,8],out_channel=[4,8,16]):
        super(Model_ericssion_one, self).__init__()
        self.in_channel = in_channel
        self.num_clase = num_clase
        self.kernal_size=3
        # self.convs = nn.ModuleList()
        self.convs = nn.Sequential()
        for i in range(len(in_channel)):
            conv_stack=nn.Sequential(
                 nn.Conv1d(in_channel[i], out_channel[i], self.kernal_size, stride=1, padding=1, bias=True,
                              padding_mode='zeros',dilation=1, device=None),
                nn.BatchNorm1d(out_channel[i]),
                 nn.ReLU(),
                 nn.MaxPool1d(kernel_size=2, stride=2))
            self.convs.append(conv_stack)
        #self.dropout = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten(1,2)
        self.fc_stack = nn.Sequential(nn.Linear(512,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,self.num_clase))
        #self.FullyConnected()
    def forward(self, x):
        conv_layer=len(self.in_channel)
        for j in range(conv_layer):
            x=self.convs[j](x)
            # print('after j th,x shape is ',x.shape)
        x=self.flatten(x)
        # print('before fc x shape is',x.shape)
        x=self.fc_stack(x)
        return x


class Model_ericssion_two(nn.Module):
    def __init__(self, num_clase, in_channel=[[2,2], [4,4], [8,8]], out_channel=[[2,4],[4,8],[8,16]]):
        super(Model_ericssion_two, self).__init__()
        self.in_channel = in_channel
        self.num_clase = num_clase
        self.kernal_size=3
        self.convs = nn.Sequential()
        for i in range(len(in_channel)):
            conv_stack = nn.Sequential(
                nn.Conv1d(in_channel[i][0], out_channel[i][0], self.kernal_size, stride=1, padding=1, bias=True,
                          padding_mode='zeros', dilation=1, device=None),
                nn.BatchNorm1d(out_channel[i][0]),
                nn.ReLU(),
                nn.Conv1d(in_channel[i][1], out_channel[i][1], self.kernal_size, stride=1, padding=1, bias=True,
                          padding_mode='zeros', dilation=1, device=None),
                nn.BatchNorm1d(out_channel[i][1]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2))
            self.convs.append(conv_stack)
        self.flatten = nn.Flatten(1, 2)
        # self.fc_stack = nn.Sequential(nn.Linear(512, 64),nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 32), nn.BatchNorm1d(32),nn.ReLU(),
        #                               nn.Linear(32, self.num_clase),nn.BatchNorm1d(num_clase),nn.ReLU())
        self.fc_stack = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 32),nn.ReLU(),
                                      nn.Linear(32, self.num_clase),nn.ReLU())#, nn.Dropout(0.3)
    def forward(self, x):
        conv_layer=len(self.in_channel)
        for j in range(conv_layer):
            x=self.convs[j](x)
            # print('after j th,x shape is ',x.shape)
        x=self.flatten(x)
        # print('after j th,x shape is ',x.shape)
        x=self.fc_stack(x)
        return x


class Model_base_ericssion(nn.Module):
    def __init__(self, num_clase, in_channel=[[2,2,2], [4,4,4], [8,8,8]], out_channel=[[2,2,4],[4,4,8],[8,8,16]]):
        super(Model_base_ericssion, self).__init__()
        self.in_channel = in_channel
        self.num_clase = num_clase
        self.kernal_size=3
        self.convs = nn.Sequential()
        for i in range(len(in_channel)):
            conv_stack = nn.Sequential(
                nn.Conv1d(in_channel[i][0], out_channel[i][0], self.kernal_size, stride=1, padding=1, bias=True,
                          padding_mode='zeros', dilation=1, device=None),
                nn.BatchNorm1d(out_channel[i][0]),
                nn.ReLU(),
                nn.Conv1d(in_channel[i][1], out_channel[i][1], self.kernal_size, stride=1, padding=1, bias=True,
                          padding_mode='zeros', dilation=1, device=None),
                nn.BatchNorm1d(out_channel[i][1]),
                nn.ReLU(),
                nn.Conv1d(in_channel[i][2], out_channel[i][2], self.kernal_size, stride=1, padding=1, bias=True,
                          padding_mode='zeros', dilation=1, device=None),
                nn.BatchNorm1d(out_channel[i][2]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2))
            self.convs.append(conv_stack)
        self.flatten = nn.Flatten(1, 2)
        # self.fc_stack = nn.Sequential(nn.Linear(512, 64),nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 32), nn.BatchNorm1d(32),nn.ReLU(),
        #                               nn.Linear(32, self.num_clase),nn.BatchNorm1d(num_clase),nn.ReLU())
        self.fc_stack = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 32),nn.ReLU(),
                                      nn.Linear(32, self.num_clase),nn.ReLU())#, nn.Dropout(0.3)
    def forward(self, x):
        conv_layer=len(self.in_channel)
        for j in range(conv_layer):
            x=self.convs[j](x)
            # print('after j th,x shape is ',x.shape)
        x=self.flatten(x)
        # print('after j th,x shape is ',x.shape)
        x=self.fc_stack(x)
        return x


class Model_base_ericssion_18layer(nn.Module):
    def __init__(self, num_clase, in_channel=[[2,2,2,2,2], [4,4,4,4,4], [8,8,8,8,8]], out_channel=[[2,2,2,2,4],[4,4,4,4,8],[8,8,8,8,16]]):
        super(Model_base_ericssion_18layer, self).__init__()
        self.in_channel = in_channel
        self.num_clase = num_clase
        self.kernal_size=3
        self.convs = nn.Sequential()
        for i in range(len(in_channel)):
            conv_stack = nn.Sequential(
                nn.Conv1d(in_channel[i][0], out_channel[i][0], self.kernal_size, stride=1, padding=1, bias=True,
                          padding_mode='zeros', dilation=1, device=None),
                nn.BatchNorm1d(out_channel[i][0]),
                nn.ReLU(),
                nn.Conv1d(in_channel[i][1], out_channel[i][1], self.kernal_size, stride=1, padding=1, bias=True,
                          padding_mode='zeros', dilation=1, device=None),
                nn.BatchNorm1d(out_channel[i][1]),
                nn.ReLU(),
                nn.Conv1d(in_channel[i][2], out_channel[i][2], self.kernal_size, stride=1, padding=1, bias=True,
                          padding_mode='zeros', dilation=1, device=None),
                nn.BatchNorm1d(out_channel[i][2]),
                nn.ReLU(),
                nn.Conv1d(in_channel[i][3], out_channel[i][3], self.kernal_size, stride=1, padding=1, bias=True,
                          padding_mode='zeros', dilation=1, device=None),
                nn.BatchNorm1d(out_channel[i][3]),
                nn.ReLU(),
                nn.Conv1d(in_channel[i][4], out_channel[i][4], self.kernal_size, stride=1, padding=1, bias=True,
                          padding_mode='zeros', dilation=1, device=None),
                nn.BatchNorm1d(out_channel[i][4]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2))
            self.convs.append(conv_stack)
        self.flatten = nn.Flatten(1, 2)
        # self.fc_stack = nn.Sequential(nn.Linear(512, 64),nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 32), nn.BatchNorm1d(32),nn.ReLU(),
        #                               nn.Linear(32, self.num_clase),nn.BatchNorm1d(num_clase),nn.ReLU())
        self.fc_stack = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 32),nn.ReLU(),
                                      nn.Linear(32, self.num_clase),nn.ReLU())#, nn.Dropout(0.3)
    def forward(self, x):
        conv_layer=len(self.in_channel)
        for j in range(conv_layer):
            x=self.convs[j](x)
            # print('after j th,x shape is ',x.shape)
        x=self.flatten(x)
        # print('after j th,x shape is ',x.shape)
        x=self.fc_stack(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            # nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv1d(2, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        # self.fc = nn.Linear(512, num_classes)#ori
        self.fc = nn.Linear(1024, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print('before fc the out shape is', out.shape)
        out = F.avg_pool2d(out, 4)
        # print('before fc the out shape is',out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

