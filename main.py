from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import data, model
from train import train_epoch, test_epoch

def save_graph_image(epoch, train_losses, train_accuracies, test_losses, test_accuracies):
    # import pdb; pdb.set_trace()
    x = np.array(range(1,epoch+1))
    acc_y1 = np.array(train_accuracies)
    acc_y2 = np.array(test_accuracies)

    loss_y1 = np.array(train_losses)
    loss_y2 = np.array(test_losses)

    plt.plot(x, acc_y1, linestyle="solid")
    plt.plot(x, acc_y2, linestyle="dashed")
    plt.plot(x, loss_y1, linestyle="dashdot")
    plt.plot(x, loss_y2, linestyle="dotted")
    # plt.legend((acc_y1[0], acc_y2[0], loss_y1[0], loss_y2[0]), ("Train Accuracy", "Test Accuracy", "Train Loss", "Test Loss"), loc=2)
    plt.savefig('graph.png')

parser = argparse.ArgumentParser(description='MNIST TRAINING')
parser.add_argument('--epochs', type=int, default=200,
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
train_losses, train_accuracies = [],[]
test_losses,  test_accuracies  = [],[]

for epoch in range(1, args.epochs + 1):
    print("Epoch: {}".format(epoch))
    train_loss, train_accuracy = train_epoch(args, net, device, train_data, optimizer, epoch)
    test_loss , test_accuracy  = test_epoch(args, net, device, test_data)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

save_graph_image(epoch, train_losses, train_accuracies, test_losses, test_accuracies)
torch.save(net.state_dict(), 'mnist_model_params.pth')
