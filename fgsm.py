import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import random

import model

def generate_adversarial_examples():
    data = np.load('./data/image.npy')
    target = np.load('./data/labels.npy')



    net = model.MnistClassifer()
    parameters = torch.load('mnist_model_params.pth')
    net.load_state_dict(parameters)


    i = random.randint(0, len(data)-1)
    x, y = torch.from_numpy(data[i]), torch.from_numpy(np.array([target[i]]))

    x = x.view(-1,1,28,28)

    y_pred = net(x)


    loss = nn.CrossEntropyLoss()(y_pred, y)
    loss.backward()

    import pdb; pdb.set_trace()

    adv_sample = np.sign(loss)


    # for i in len(adv_samples()):
    pil_img = Image.fromarray(adv_sample)
    pil_img.save('test.png')



if __name__ == '__main__':
    generate_adversarial_examples()
