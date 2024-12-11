#!/usr/bin/env python
# coding: utf-8

# name = "k-PPCAs"
# version = "0.0.1"
# authors = [
#   {name="Ke Han", email="kh19r@fsu.edu"},
#   {name="Adrian Barbu", email="abarbu@fsu.edu"},
# ]

import torch
import torch.nn.functional as F
from scipy.io import savemat

import gc

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import clip
from torch.utils.data import Dataset, TensorDataset, DataLoader


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def data_augmentation(n_px):
    return Compose([
        transforms.RandomResizedCrop(size=n_px, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        _convert_image_to_rgb,  # from _transform
        ToTensor(),  # from _transform
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def features32(net, x):
    x = x.type(net.conv1.weight.dtype)
    for conv, bn in [(net.conv1, net.bn1), (net.conv2, net.bn2), (net.conv3, net.bn3)]:
        x = net.relu(bn(conv(x)))
    x = net.avgpool(x)
    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)
    x = F.avg_pool2d(x, x.shape[2])
    return x


def features(net, x):
    x = x.type(net.conv1.weight.dtype)
    for conv, bn in [(net.conv1, net.bn1), (net.conv2, net.bn2), (net.conv3, net.bn3)]:
        x = net.relu(bn(conv(x)))
    x = net.avgpool(x)
    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)
    x = net.attnpool(x)  # 288
    return x.detach().clone()


def features_ViT(net, x):
    x = x.type(net.conv1.weight.dtype)
    x = net(x)
    return x.detach().clone()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    nc = 100

    # nx = 288 # for CLIP-ResNet
    nx = 224  # for ResNet and CLIP-ViT

    # print(clip.available_models())
    # model, preprocess = clip.load('RN50x4', device)
    # model, preprocess = clip.load('ViT-B/32', device)
    model, preprocess = clip.load('ViT-L/14', device)

    train = 1
    if train:
        CIFAR100_addr = "F:/LargeDataFile/CIFAR100/cifar-100-python/train"
        CIFAR100_dict = unpickle(CIFAR100_addr)
    else:
        CIFAR100_addr = "F:/LargeDataFile/CIFAR100/cifar-100-python/test"
        CIFAR100_dict = unpickle(CIFAR100_addr)
    x_raw = CIFAR100_dict['data']
    y = CIFAR100_dict['fine_labels']

    n = x_raw.shape[0]
    x_3channels = x_raw.reshape(n, 3, 32, 32)
    n_half = int(n / 2)
    print(n_half)
    for half in range(2):
        x = torch.zeros([n_half, 3, nx, nx], dtype=torch.float32)
        n_start = n_half * half
        n_end = n_half * (half + 1)
        for i in range(n_half):
            im = Image.fromarray(x_3channels[i + n_start].transpose(1, 2, 0), 'RGB')
            x[i, :, :, :] = _transform(nx)(im)
            if (i + 1) % 1000 == 0:
                print(i + n_start)
        data = TensorDataset(x)

        batchsize = 100
        loader = DataLoader(data, batch_size=batchsize, shuffle=False, drop_last=False)
        i = 0
        with torch.no_grad():
            for images in loader:
                images = images[0].to(device)

                # fi=features(model.visual,images)
                # fi=features32(model.visual,images)
                fi = features_ViT(model.visual, images)

                if i == 0 and half == 0:
                    X = fi.clone()
                else:
                    X = torch.cat((X, fi.clone()), dim=0)
                i += 1
                if i % 10 == 0:
                    print(i * 100)
        del x, data, loader
        gc.collect()
    p = X.shape[1]
    print(X.shape)
    if train:
        # name='F:/LargeDataFile/CIFAR100/CIFAR100_ViT/train_ViT-B32_{}_ImageNetNorm.mat'.format(nx)
        name = 'F:/LargeDataFile/CIFAR100/CIFAR100_ViT/train_ViT-L14_{}_ImageNetNorm.mat'.format(nx)
        savemat(name, {'feature': X.float().cpu().numpy(), 'label': y})
    else:
        # name='F:/LargeDataFile/CIFAR100/CIFAR100_ViT/val_ViT-B32_{}_ImageNetNorm.mat'.format(nx)
        name = 'F:/LargeDataFile/CIFAR100/CIFAR100_ViT/val_ViT-L14_{}_ImageNetNorm.mat'.format(nx)
        savemat(name, {'feature': X.cpu().float().numpy(), 'label': y})
