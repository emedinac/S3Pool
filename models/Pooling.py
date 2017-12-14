import torch
import torch.nn as nn
from torch.autograd import Variable

class StochasticPool2DLayer(nn.Module):
    def __init__(self, pool_size=2, maxpool=True, training=False, grid_size=None, **kwargs):
        super(StochasticPool2DLayer, self).__init__(**kwargs)
        self.rng = torch.cuda.manual_seed_all(123) # this changed in Pytorch for working
        self.pool_size = pool_size
        self.maxpool_flag = maxpool
        self.training = training
        if grid_size:
            self.grid_size = grid_size
        else:
            self.grid_size = pool_size

        self.Maxpool = torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=1)
        self.Avgpool = torch.nn.AvgPool2d(kernel_size=self.pool_size,
                                          stride=self.pool_size,
                                          padding=self.pool_size//2,)
        self.padding = nn.ConstantPad2d((0,1,0,1),0)

    def forward(self, x, training=False, **kwargs):
        if self.maxpool_flag:
            x = self.Maxpool(x)
            x = self.padding(x)
        if not self.training:
            # print(x.size())
            x = self.Avgpool(x)
            return x
            # return x[:, :, ::self.pool_size, ::self.pool_size]       
        else:
            w, h = x.data.shape[2:]
            n_w, n_h = w//self.grid_size, h//self.grid_size
            n_sample_per_grid = self.grid_size//self.pool_size
            # print('===========================')
            idx_w = []
            idx_h = []
            if w>2 and h>2:
                for i in range(n_w):
                    offset = self.grid_size * i
                    if i < n_w - 1:
                        this_n = self.grid_size
                    else:
                        this_n = x.data.shape[2] - offset
                    
                    this_idx, _ = torch.sort(torch.randperm(this_n)[:n_sample_per_grid])
                    idx_w.append(offset + this_idx)
                #     print(i,offset,this_n)
                #     print(this_idx)
                # print('***************************************')
                # print('***************************************')
                # print('***************************************')
                # print('***************************************')
                # print('***************************************')
                # print(idx_w)
                # print('///////////////////////////////////////')
                for i in range(n_h):
                    offset = self.grid_size * i
                    if i < n_h - 1:
                        this_n = self.grid_size
                    else:
                        this_n = x.data.shape[3] - offset
                    this_idx, _ = torch.sort(torch.randperm(this_n)[:n_sample_per_grid])

                    idx_h.append(offset + this_idx)
                idx_w = torch.cat(idx_w, dim=0)
                idx_h = torch.cat(idx_h, dim=0)
            else:
                idx_w = torch.LongTensor([0])
                idx_h = torch.LongTensor([0])

            output = x[:, :, idx_w.cuda()][:, :, :, idx_h.cuda()]
            return output
