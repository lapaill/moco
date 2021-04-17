# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random

import torch
from PIL import ImageFilter


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(torch.nn.Module):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)


class TwoCropsAndTransform:
    """Take two random crops of one image as the query and key.
    If randomcrop = False, creates the same two crops, otherwise create two different crops which are constrained to be close with the proximity parameter (size > proximity)
    """

    def __init__(self, base_transform, randomcrop=False, original_size=256, size=96, proximity=48):
        self.base_transform = base_transform
        self.randomcrop = randomcrop
        self.original_size = original_size
        self.size = size
        self.proximity = proximity

    def __call__(self, x):
        if self.randomcrop == True:
            y = x.copy()
            rand1 = int(torch.randint(0, self.original_size - self.size, (1,)))
            rand2 = int(torch.randint(0, self.original_size - self.size, (1,)))
            x = x.crop((rand1, rand2, rand1 + self.size, rand2 + self.size))
            y = y.crop((max(0, rand1+self.proximity-self.size), max(0, rand2+self.proximity-self.size), min(
                self.original_size, rand1-self.proximity+2*self.size), min(self.original_size, rand2-self.proximity+2*self.size)))
            xdim, ydim = y.size
            rand3 = int(torch.randint(0, xdim - self.size, (1,)))
            rand4 = int(torch.randint(0, ydim - self.size, (1,)))
            y = y.crop((rand3, rand4, rand3 + self.size, rand4 + self.size))
            q = self.base_transform(y)
            k = self.base_transform(x)
            return [q, k]
        else:
            rand1 = int(torch.randint(0, self.original_size - self.size, (1,)))
            rand2 = int(torch.randint(0, self.original_size - self.size, (1,)))
            x = x.crop((rand1, rand2, rand1 + self.size, rand2 + self.size))
            q = self.base_transform(x)
            k = self.base_transform(x)
            return [q, k]
