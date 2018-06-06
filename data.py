import torch
from torchvision import datasets, transforms


def train_loader(args, kwargs):
    train_data = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_data

def test_loader(args, kwargs):
    test_data = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return test_data
