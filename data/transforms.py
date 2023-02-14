# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class ToTensor(object):
    def __call__(self, lines):
        return lines


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, lines):
        lines = F.normalize(lines,mean=self.mean, std=self.std)
        return lines


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lines):
        for t in self.transforms:
            lines = t(lines)
        return lines

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
