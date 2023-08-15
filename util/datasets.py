# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pickle
import matplotlib.pyplot as plt
import numpy as np

def is_valid_file_fn(path_):
    return 'pkl' in path_


def load_fn(path_):
    with open(path_, 'rb') as handle:
        dict_ = pickle.load(handle)

    img_ = dict_["img"]
    recon_ = dict_["recon"]
    diff = np.abs((img_ - recon_))[:, :, 0]

    # plt.subplot(1, 3, 1)
    # plt.imshow(img_, cmap='gray')
    # plt.subplot(1, 3, 2)
    # plt.imshow(recon_, cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(diff, cmap='gray')
    # plt.show()

    return PIL.Image.fromarray(diff)



def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform, loader=load_fn, is_valid_file=is_valid_file_fn)
    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = [0]  # IMAGENET_DEFAULT_MEAN
    std = [1]  # IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            # auto_augment=args.aa,
            interpolation='bicubic', #transforms.InterpolationMode.BICUBIC,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    # if args.input_size <= 224:
    #     crop_pct = 224 / 256
    # else:
    #     crop_pct = 1.0
    # size = int(args.input_size / crop_pct)
    # t.append(
    #     transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    # )

    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
