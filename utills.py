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
