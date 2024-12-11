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
from os import listdir
from os.path import isfile, join

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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # nx = 288
    nx = 224

    train_folder = 'F:/LargeDataFile/ILSVRC/Data/MINI-ImageNet/train/'
    names = listdir(train_folder)

    filename_cls_map = {}
    cls_filename_map = {}
    for cls, name in enumerate(names):
        filename_cls_map[name] = cls
        cls_filename_map[cls] = name

    nc = len(names)
    print(nc)

    classes = torch.load('F:/LargeDataFile/ILSVRC/classes_val.pth')

    num_augmented = 10

    # print(clip.available_models())
    # model, preprocess = clip.load('RN50x4', device)
    # model, preprocess = clip.load('RN50', device)
    # model, preprocess = clip.load('ViT-B/32', device)
    model, preprocess = clip.load('ViT-L/14', device)

    train = 1
    for j in range(91, nc):
        if train:
            path = 'F:/LargeDataFile/ILSVRC/Data/MINI-ImageNet/train/' + names[j]
            files = [f for f in listdir(path) if isfile(join(path, f))]
        else:
            path = 'F:/LargeDataFile/ILSVRC/Data/MINI-ImageNet/val/' + names[j]
            files = [f for f in listdir(path) if isfile(join(path, f))]

        n_files = len(files)
        n_imgs = n_files * num_augmented
        x_aug = torch.zeros([n_imgs, 3, nx, nx], dtype=torch.float32)
        for i in range(n_files):
            im = Image.open(join(path, files[i]))
            for aug_i in range(num_augmented):
                x_aug[i * num_augmented + aug_i, :, :, :] = data_augmentation(nx)(im)
        data = TensorDataset(x_aug)
        loader = DataLoader(data, batch_size=num_augmented, shuffle=False, drop_last=False)

        i = 0
        for images in loader:
            images = images[0].to(device)
            with torch.no_grad():
                # fi = features(model.visual, images)
                # fi = features32(model.visual, images)
                fi = features_ViT(model.visual, images)
            if i == 0:
                X = fi
            else:
                X = torch.cat((X, fi), dim=0)
            i += 1
        p = X.shape[1]
        print(X.shape)
        if train:
            # name = 'F:/LargeDataFile/ILSVRC/miniImgNet/miniImgNet_ViT/train_aug_ViT-B32_224_ImageNetNorm/%s.mat' % (names[j])
            name = 'F:/LargeDataFile/ILSVRC/miniImgNet/miniImgNet_ViT/train_aug_ViT-L14_224_ImageNetNorm/%s.mat' % (
            names[j])
            savemat(name, {'feature': X.float().cpu().numpy(), 'label': filename_cls_map[names[j]]})

        else:  # not needed to test augmented val
            print('skip augmented val')
        print(j, name)
