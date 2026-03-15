# This file is only for reference and need not be run or modified.

import numpy as np
import torchvision
from einops import rearrange

def shuffle(patch_size: int, data_point):
    # Image of size (img_dim x img_dim), img_dim should be divisible by patch_size, and data shape should be channel-last HxWxC
    height_reshape= int(np.shape(data_point)[0]/patch_size)
    img = rearrange(data_point, '(h s1) (w s2) c -> (h w) s1 s2 c', s1=patch_size, s2=patch_size)
    np.random.shuffle(img)
    img = rearrange(img, '(h w) s1 s2 c -> (h s1) (w s2) c', h=height_reshape, s1=patch_size, s2=patch_size)
    return img


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

def create_patch_shuffled_data(dataset: torchvision.datasets, file_name: str, patch_size: int = 16)-> None:
    images = []
    labels = []
    for i, l in dataset:
        m = np.asarray(i)
        m = shuffle(patch_size = patch_size, data_point = m)
        images.append(m)
        labels.append(l)
    np.savez(file = file_name, data = images, labels = labels)

np.random.seed(1)
create_patch_shuffled_data(testset, file_name= "test_patch_16", patch_size = 16)
create_patch_shuffled_data(testset, file_name= "test_patch_8", patch_size = 8)