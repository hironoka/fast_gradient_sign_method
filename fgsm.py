import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from PIL import Image
import random

import model



def save_image(x, filename):
    img = transforms.ToPILImage()(x.view(-1,28,28))
    img = img.convert('RGB')
    img.save(filename)

    
def generate_adversarial_examples(args):
    eps = args.eps

    # data
    data, target = np.load('./data/image.npy'), np.load('./data/labels.npy')

    i = random.randint(0, len(data)-1)
    x, y = torch.from_numpy(data[i]).view(-1,1,28,28), torch.from_numpy(np.array([target[i]]))
    x.requires_grad=True


    # model
    net = model.MnistClassifer()
    param = torch.load('mnist_model_params.pth')
    net.load_state_dict(param)

    y_pred = net(x)
   
    creterion = nn.CrossEntropyLoss()
    loss = creterion(y_pred, y)
    loss.backward()

    # fast gradient sign method
    eta = eps * np.sign(x.grad.view(-1))
    adv = x.data + eta.view(-1,1,28,28)

    adv_y_pred = net(adv)

    save_image(x, 'x.png')
    save_image(x.grad, 'noize.png')
    save_image(adv, 'adv_sample.png')
    print('y: ', int(y), ' y_pred: ', int(y_pred.max(1)[1]), ' adv_y_pred: ', int(adv_y_pred.max(1)[1]))
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST TRAINING')
    parser.add_argument('--eps', type=int, default=0.001,
                        help='Epsilon')
    args = parser.parse_args()


    generate_adversarial_examples(args)
