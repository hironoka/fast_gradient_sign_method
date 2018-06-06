
# ADVERSARIAL EXAMPLES IN THE PHYSICAL WORLD
# https://arxiv.org/pdf/1607.02533.pdf

# ADVERSARIAL MACHINE LEARNING AT SCALE
# https://arxiv.org/pdf/1611.01236.pdf


import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import math

import model


def save_image(x, filename):
    img = transforms.ToPILImage()(x.view(-1,28,28))
    img = img.convert('RGB')
    img.save(filename)


def basic_iterative_method(args):
    eps = args.eps
    alpha = atgs.alpha
    n = math.ceil([e + 4, e * 1.25].min(1))

    # model
    net = model.MnistClassifer()
    param = torch.load('mnist_model_params.pth', map_location='cpu')
    net.load_state_dict(param)


    # basic iterative method

    # data
    for i in range(n):
        data, target = np.load('./data/image.npy'), np.load('./data/labels.npy')
        i = random.randint(0, len(data)-1)
        x, y = torch.from_numpy(data[i]).view(-1,1,28,28), torch.from_numpy(np.array([target[i]]))
        x.requires_grad=True

        y_pred = net(x)

        creterion = nn.CrossEntropyLoss()
        loss = creterion(y_pred, y)
        loss.backward()

        eta = alpha * np.sign(x.grad.view(-1))
        adv = x.data + eta.view(-1,1,28,28)

        adv_y_pred = net(adv)

    save_image(x, 'result/x.png')
    save_image(x.grad, 'result/noize.png')
    save_image(adv, 'result/adv_sample.png')
    print('y: ', int(y), ' y_pred: ', int(y_pred.max(1)[1]), ' adv_y_pred: ', int(adv_y_pred.max(1)[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST TRAINING')
    parser.add_argument('--eps', type=int, default=0.001,
                        help='Epsilon')
    parser.add_argument('--alpha', type=int, default=1,
                        help='Alpha')
    args = parser.parse_args()


    basic_iterative_method(args)
