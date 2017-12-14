'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from .Pooling import StochasticPool2DLayer# , ZeilerPool2DLayer

import torch.nn.functional as F

cfg_vgg = {
    '6ls1': [32, 32, 'S', 64, 64, 'S', 128, 128],
    '6ls2': [64, 64, 'S', 128, 128, 'S', 256, 256],
    '6ls3': [128, 128, 'S', 256, 256, 'S', 512, 512,],
    '6ls4': [128, 128, 'S', 256, 256, 'S', 512, 512,'S'],
    'VGG19_S3': [64, 64, 'S', 128, 128, 'S', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfg_res = {
    'ref': [32, 32, 'S', 64, 'S', 128, 10],
}


# With Global Averga Pooling
class CustomVGG(nn.Module):
    def __init__(self, vgg_name, grids=None):
        super(Custom, self).__init__()
        if grids:
            self.grid_sizes = grids

        self.features = self._make_layers(cfg_vgg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.dropout = nn.Dropout(p=0.2, inplace=True)

    def forward(self, x, training=False):
        self.training = training
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg_vgg):
        layers = []
        in_channels = 3
        grid_cnt = 0
        for x in cfg_vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'S':
                
                layers += [StochasticPool2DLayer(pool_size=2, maxpool=True, training=self.training, grid_size=self.grid_sizes[grid_cnt]),
                            ]
                grid_cnt += 1
            elif x == 'Z':
                layers += [ZeilerPool2DLayer(net, pool_size=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            # nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True),
                            # nn.Dropout2d(p=0.2, inplace=False),
                            ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=8, stride=8)]
        return nn.Sequential(*layers)









class CustomBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(CustomBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CustomBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(CustomBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 'ref': [32, 32, 'S', 64, 'S', 128],

class CustomResNet(nn.Module):
    def __init__(self, name, block, num_blocks, grids=None):
        super(CustomResNet, self).__init__()
        if grids:
            self.grid_sizes = grids
        self.conv1 = nn.Conv2d(3, cfg_res[name][0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.features = self._make_layers(cfg_res[name], block, num_blocks)

    def _make_layers(self, Topology, block, num_blocks):
        layers = []
        in_channels = Topology[0]
        grid_cnt = 0
        ind_num_blocks = 0
        for x in Topology[1:-1]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'S':
                
                layers += [StochasticPool2DLayer(pool_size=2, maxpool=True, training=self.training, grid_size=self.grid_sizes[grid_cnt]),
                            ]
                grid_cnt += 1
            elif x == 'Z':
                layers += [ZeilerPool2DLayer(net, pool_size=2)]
            else:
                for i in range(num_blocks[ind_num_blocks]):
                    layers += [  block(in_channels, x)  ]
                    in_channels = x
                    x = x * block.expansion

                ind_num_blocks += 1
        layers += [nn.Conv2d(in_channels, Topology[-1], kernel_size=1, stride=1, padding=1)]
        layers += [nn.AvgPool2d(kernel_size=8, stride=8)]
        return nn.Sequential(*layers)


    def forward(self, x, training=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.features(out)
        out = out.view(out.size(0), -1)
        
        return out




def RefPaperS3Pool():
    return CustomResNet('ref',CustomBasicBlock, [3,3,3], [16,8]) # NICE!!! CustomBasicBlock implementation.

# def ResNet18():
#     return ResNet(BasicBlock, [2,2,2,2])

# def ResNet34():
#     return ResNet(BasicBlock, [3,4,6,3])

# def ResNet50():
#     return ResNet(Bottleneck, [3,4,6,3])

# def ResNet101():
#     return ResNet(Bottleneck, [3,4,23,3])

# def ResNet152():
#     return ResNet(Bottleneck, [3,8,36,3])