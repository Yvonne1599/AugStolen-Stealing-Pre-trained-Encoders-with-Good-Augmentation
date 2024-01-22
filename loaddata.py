# This code is for the consine method
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torch.utils.data import DataLoader
from resnet_wider import resnet50x1
from torchvision import models
from PIL import Image
import time
from network import C10
from torch.utils.data import Dataset
import random
from noise_utils import *

# This part loads different datasets with the use of torchvision.datasets.


def load_data(data, bs, aug=False):

    # Seem all datasets own 10 classes.
    transform_ = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()])
    # Other data augmentation methods to enhance the attack.
    transform_aug = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.ToPILImage(),
            AddGaussianNoise(0, 1e-4, 1e-4),
            transforms.ToTensor()
        ])

    if aug:
        transform = transform_aug
    else:
        transform = transform_

    if data == 'cifar10':
        train_dataset = datasets.CIFAR10(
            './dataset/cifar10/', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(
            './dataset/cifar10/', train=False, download=True, transform=transform)

    elif data == 'cifar10-2k':
        train_dataset = datasets.CIFAR10(
            './dataset/cifar10/', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(
            './dataset/cifar10/', train=False, download=True, transform=transform)
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [20000, len(train_dataset)-20000])

    elif data == 'stl10':
        train_dataset = datasets.STL10(
            './dataset/stl10', split="train", download=True, transform=transform)
        test_dataset = datasets.STL10(
            './dataset/stl10', split="test", download=True, transform=transform)

    elif data == 'stl10u':
        train_dataset = datasets.STL10(
            './dataset/stl10', split="unlabeled", download=True, transform=transform)
        test_dataset = datasets.STL10(
            './dataset/stl10', split="test", download=True, transform=transform)

    elif data == 'stl10-50k':
        dataset = datasets.STL10(
            './dataset/stl10', split="unlabeled", download=True, transform=transform)
        indices = list(range(len(dataset)))
        random.seed(310)
        random.shuffle(indices)
        train_dataset = torch.utils.data.Subset(dataset, indices[:50000])
        test_dataset = torch.utils.data.Subset(dataset, indices[50000:])

    elif data == 'gtsrb':
        train_dataset = datasets.ImageFolder(
            './dataset/GTSRB/Train/', transform=transform)
        test_dataset = datasets.ImageFolder(
            './dataset/GTSRB/Train/', transform=transform)
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [39000, len(train_dataset)-39000])

    elif data == 'imagenet':
        train_dataset = datasets.ImageFolder(
            './dataset/imagenet/train/', transform=transform)
        test_dataset = datasets.ImageFolder(
            './dataset/imagenet/train/', transform=transform)
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [20000, len(train_dataset)-20000])

    elif data == 'mnist':
        train_dataset = datasets.MNIST(
            root="./dataset/MNIST/", transform=transform, train=True, download=True)
        test_dataset = datasets.MNIST(
            root="./dataset/MNIST/", transform=transform, train=False)

    elif data == 'fashion-mnist':
        train_dataset = datasets.FashionMNIST(
            root="./dataset/fashionmnist/", transform=transform, train=True, download=True)
        test_dataset = datasets.FashionMNIST(
            root="./dataset/fashionmnist/", transform=transform, train=False)

    print('dataset: ', len(train_dataset))
    print('data augmentation: ', aug)

    train_loader = DataLoader(
        train_dataset, batch_size=bs, num_workers=5, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs,
                             num_workers=5, drop_last=False, shuffle=False)

    return train_loader, test_loader
