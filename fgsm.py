import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import random

import model

def generate_adversarial_examples(args):
    eps = args.eps

    data, target = np.load('./data/image.npy'), np.load('./data/labels.npy')

    net = model.MnistClassifer()
    param = torch.load('mnist_model_params.pth')
    net.load_state_dict(param)

    # 適当に1枚選ぶ
    i = random.randint(0, len(data)-1)
    x = torch.from_numpy(data[i]).view(-1,1,28,28)
    y = torch.from_numpy(np.array([target[i]]))

    y_pred = net(x)

    print('y: ', int(y), ' y_pred: ', int(y_pred.max(1)[1]))

    creterion = nn.CrossEntropyLoss()
    loss = creterion(y_pred, y)
    loss.backward()

    adv_sample = x.view(28,28).numpy() + eps * np.sign(x.data.view(28,28).numpy())

    pil_img = Image.fromarray(adv_sample)
    if pil_img.mode != "RGB":
        pil_img= pil_img.convert("RGB")
    pil_img.save('test.png')




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MNIST TRAINING')
    parser.add_argument('--eps', type=int, default=0.01,
                        help='Epsilon')
    args = parser.parse_args()
    

    generate_adversarial_examples(args)
