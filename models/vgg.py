'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from .Pooling import StochasticPool2DLayer# , ZeilerPool2DLayer

cfg = {
    'VGG11_S3': [64, 'S', 128, 'S', 256, 256, 'S', 512, 512, 'S', 512, 512, 'S'],
    'VGG19_S3': [64, 64, 'S', 128, 128, 'S', 256, 256, 256, 256, 'S', 512, 512, 512, 512, 'S', 512, 512, 512, 512, 'S'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier1 = nn.Linear(512, 4096)
        self.classifier2 = nn.Linear(4096, 4096)
        self.classifier3 = nn.Linear(4096, 100)
        self.dropout = nn.Dropout(p=0.2, inplace=True)

    def forward(self, x):
        out = self.features(x)
        out = out.view( -1, out.size(1)*out.size(2)*out.size(3) )
        out = self.dropout(out)
        out = self.classifier1(out)
        out = self.dropout(out)
        out = self.classifier2(out)
        out = self.dropout(out)
        out = self.classifier3(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'S':
                
                layers += [StochasticPool2DLayer(pool_size=2, maxpool=True, grid_size=self.grid_sizes[grid_cnt]),
                            # nn.ConvTranspose2d(in_channels, 128, kernel_size=5, padding=2),
                            # nn.BatchNorm2d(128),
                            # nn.Conv2d(128, in_channels, kernel_size=5, padding=2),
                            # nn.BatchNorm2d(in_channels),
                            ]
                grid_cnt += 1
            elif x == 'Z':
                layers += [ZeilerPool2DLayer(net, pool_size=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True),
                            nn.Dropout2d(p=0.2, inplace=False),
                            ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# With Global Averga Pooling
class VGG_GAP(nn.Module):
    def __init__(self, vgg_name, grids=None):
        super(VGG_GAP, self).__init__()
        if grids:
            self.grid_sizes = grids

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 100)
        self.dropout = nn.Dropout(p=0.2, inplace=True)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        grid_cnt = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'S':
                
                layers += [StochasticPool2DLayer(pool_size=2, maxpool=True, grid_size=self.grid_sizes[grid_cnt]),
                            # nn.ConvTranspose2d(in_channels, 128, kernel_size=5, padding=2),
                            # nn.BatchNorm2d(128),
                            # nn.Conv2d(128, in_channels, kernel_size=5, padding=2),
                            # nn.BatchNorm2d(in_channels),
                            ]
                grid_cnt += 1
            elif x == 'Z':
                layers += [ZeilerPool2DLayer(net, pool_size=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True),
                            nn.Dropout2d(p=0.2, inplace=False),
                            ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
