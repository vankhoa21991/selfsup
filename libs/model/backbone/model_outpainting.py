# Basile Van Hoorick, Jan 2020

import copy
import cv2
import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import scipy
import shutil
import skimage
import time
import torch
import torch.nn.functional as F
import torchvision
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict, OrderedDict
from html4vision import Col, imagetable
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from skimage import io
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models, utils
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CEGenerator(nn.Module):
    def __init__(self, channels=3, extra_upsample=False):
        super(CEGenerator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        if not (extra_upsample):
            self.model = nn.Sequential(
                *downsample(channels, 64, normalize=False),
                *downsample(64, 64),
                *downsample(64, 128),
                *downsample(128, 256),
                *downsample(256, 512),
                nn.Conv2d(512, 4000, 1),
                *upsample(4000, 512),
                *upsample(512, 256),
                *upsample(256, 128),
                *upsample(128, 64),
                nn.Conv2d(64, channels, 3, 1, 1),
                nn.Tanh()
            )
        else:
            self.model = nn.Sequential(
                *downsample(channels, 64, normalize=False),
                *downsample(64, 64),
                *downsample(64, 128),
                *downsample(128, 256),
                *downsample(256, 512),
                nn.Conv2d(512, 4000, 1),
                *upsample(4000, 512),
                *upsample(512, 256),
                *upsample(256, 128),
                *upsample(128, 64),
                *upsample(64, 64),
                nn.Conv2d(64, channels, 3, 1, 1),
                nn.Tanh()
            )

    def forward(self, x):
        count = 0
        for layer in self.model:
            x = layer(x)
            if count == 14:
                encoded = x  # (4000,16,16)
            count = count + 1

        # print(encoded.size())
        return x, encoded.view(-1, 4000*6*6)


class CEDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super(CEDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)






