# (based on skeleton code for CSCI-B 657)

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np

class PatchShuffled_CIFAR10(Dataset):
    def __init__(self, data_file_path = 'test_patch_16.npz', transforms = None):
        super(PatchShuffled_CIFAR10, self).__init__()
        with np.load(data_file_path) as k:
           self.images = k['data']
           self.labels = k['labels']
        self.transform = transforms
  
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, label = self.images[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.as_tensor(label)