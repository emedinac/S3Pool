'''Train CIFAR10 with PyTorch.'''
# from __future__ import print_function
from termcolor import colored, cprint

import os
import gzip # to decompress
import _pickle as pickle# to serialize objects
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar
from torch.autograd import Variable

# import matplotlib.pyplot as plt

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0" # USED ONLY IF torch.cuda.device_count() IS NOT USED FOR INITIAL CONFIGURATION.
# torch.cuda.set_device(0)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lrsch', default=["50","100",], nargs='+', type=str, help='learning rate schedule')
parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = start_epoch = 0
lr = args.lr
schedule = [int(i) for i in args.lrsch[0].split(',')]
print("learning rate:", lr)

###############################################
################################################ Data and model name
###############################################
namemodel='semiVGG_6l1_2S_10'
namemodel='ref_resnet'
description =  ['S3pool in [64, 64, S, 128, 128, S, 256, 256] for CIFAR10 database \n',
                'optim.SGD(net.parameters(), lr=lr) \n',
                'run main.py --lr=1 --lrsch=80,130 --epoch=150 \n',
                'no L2, no dropout in linear, no batchnorm, using GAP, using data augmentation ']
description =  ['paper reference',
                'run main.py --lr=0.1 --lrsch=50,110,135 --epoch=150',
                ]
# namemodel = 'VGG16'
# description = 'Model employed for Deep Compression'
###############################################
###############################################
###############################################
DATA = 'CIFAR10'
print('==> Preparing data..')
if DATA=='CIFAR10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=8)

else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=8)

print(trainset.data.shape)
print(np.mean(trainset.data, axis=(0,1,2)))
print(np.std(trainset.data, axis=(0,1,2)))

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+namemodel+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG_GAP('VGG19_S3',[16,8,4,2,1])
    # net = CustomVGG('6ls3',[16,8,4,2,1])
    # net = CustomResnet('6ls3',[16,8,4,2,1])
    net = Network((3,32,32))
    # net = VGG('VGG19')
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight.data)

net.apply(weights_init) # apply weight init Xavier or He


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    tr_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs, training=True)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        tr_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().cpu()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (tr_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return tr_loss/(batch_idx+1), 100.*correct/total
    
def test(epoch):
    global best_acc, all_weights
    net.eval()
    with torch.no_grad():
        te_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs, training=False)
            loss = criterion(outputs, targets)

            te_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum().cpu()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (te_loss/(batch_idx+1), 100.*correct/total, correct, total))

    weights_values = 0
    for param in net.parameters():
        weights_values += np.abs(param.cpu().data.numpy()).sum()
    all_weights.append(weights_values)
    if np.isnan(weights_values): # depening on GPU model, S3Pool generates possible problems in Pytorch 0.1.x
        print(  colored('=========================================================================================\n','red'), 
                colored('=========================================================================================\n ','red'), 
                colored(weights_values,'red'),'\n',
                colored('=========================================================================================\n','red'), 
                colored('=========================================================================================\n','red'),
                )
    else:
        print('(debugging...) sum the absolute weight values (L1):  ', colored(weights_values,'yellow'))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print(colored('Saving..','green'))
        state = { 'net': net.module if use_cuda else net, 'acc': acc, 'epoch': epoch }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+namemodel+'.t7')
        best_acc = acc
    return te_loss/(batch_idx+1), 100.*correct/total

def Schedule(): # Learning rate schedule
    global optimizer, lr
    lr = lr/10
    print(colored("learning rate:", 'blue'), colored(lr,'blue'))
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

all_weights = []
# VGG19_GAP_BN_drop_100:    run main.py --lr=0.1 --lrsch=40,80,110,140 --epoch=150  Loss: 0.359 | Acc: 88.120%  \  Loss: 0.320 | Acc: 89.990%


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

tr_loss_list = []; tr_acc_list = []; te_loss_list = []; te_acc_list = [];  
for epoch in range(start_epoch, start_epoch+args.epoch):
    if epoch in schedule:
        Schedule()

    tr_loss, train_acc = train(epoch)
    te_loss , test_acc  = test(epoch)

    tr_loss_list.append(tr_loss); tr_acc_list.append(train_acc); te_loss_list.append(te_loss); te_acc_list.append(test_acc)

pickle.dump((tr_loss_list, tr_acc_list, te_loss_list, te_acc_list, all_weights, description), gzip.open('./checkpoint/'+namemodel+'.p.gz', 'wb'))

# tr_loss, tr_acc, te_loss, te_acc, description = pickle.load(gzip.open('./checkpoint/VGG19_GAP_BN_drop_100.p.gz', 'rb'))

