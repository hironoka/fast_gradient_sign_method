from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import data, model
from train import train_epoch, test_epoch

parser = argparse.ArgumentParser(description='MNIST TRAINING')
parser.add_argument('--epochs', type=int, default=10,
                    help='Epoch')
parser.add_argument('--gpu', type=bool, default=False,
                    help='Use gpu or not')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--test_batch_size', type=int, default=1000,
                    help='batch size')
parser.add_argument('--solve', type=str, default='q1',
                    help='the question to solve')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#######################

# dataset
train_data, test_data = data.train_loader(args, kwargs), data.test_loader(args, kwargs)

# model
net = model.MnistClassifer().to(device)
print(net)

# optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr)
# optimizer = optim.Adam(net.parameters(), lr = 0.0001)

# train
for epoch in range(1, args.epochs + 1):
    print("Epoch: {}".format(epoch))
    train_epoch(args, net, device, train_data, optimizer, epoch)
    test_epoch(args, net, device, test_data)

torch.save(net.state_dict(), 'mnist_model_params.pth')
