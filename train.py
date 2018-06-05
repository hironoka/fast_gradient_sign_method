import torch
# from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm



def train_epoch(args, net, device, train_data, optimizer, epoch):
    net.train()
    losses = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_data)):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        y_pred = net(data)

        loss = nn.CrossEntropyLoss()(y_pred, target)
        #謎，格好の位置おかしくない

        loss.backward()
        optimizer.step()

        losses += loss
        pred = y_pred.max(1, keepdim=True)[1] # get the index of the max
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss = losses / batch_idx
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        losses / batch_idx, correct, len(train_data.dataset),
        100. * correct / len(train_data.dataset))
    )

def test_epoch(args, net, device, test_data):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up
            pred = output.max(1, keepdim=True)[1] # get the index of the max
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_data.dataset)
    print('Test set:  Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data.dataset),
        100. * correct / len(test_data.dataset)))
