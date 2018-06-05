import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistClassifer(nn.Module):
    def __init__(self):
        super(MnistClassifer, self).__init__()
        self.conv1d = nn.Conv2d(1, 10, 5)
        self.conv2d = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1d(x), 2))
        x = F.relu(F.max_pool2d(self.conv2d(self.conv2_drop(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        output = F.log_softmax(x)
        # output = nn.LogSoftmax(x)
        return output
