import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt



def train_epoch(args, net, device, train_data, optimizer, epoch):
    net.train()
    losses = 0
    correct = 0
    num = len(train_data.dataset)

    for batch_idx, (data, target) in enumerate(tqdm(train_data)):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        y_pred = net(data)

        creterion = nn.CrossEntropyLoss()
        loss = creterion(y_pred, target)
        loss.backward()
        optimizer.step()

        losses += loss
        pred = y_pred.max(1)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        losses / batch_idx, correct, num, 100. * correct / num))

    return float(losses / batch_idx), 100. * correct / num


def test_epoch(args, net, device, test_data):
    net.eval()
    losses = 0
    correct = 0
    num = len(test_data.dataset)

    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            y_pred = net(data)

            creterion = nn.CrossEntropyLoss()
            loss = creterion(y_pred, target)
            losses += loss
            pred = y_pred.max(1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Test set:  Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        losses / num , correct, num, 100. * correct / num))

    return float(losses/num), 100. * correct / num
