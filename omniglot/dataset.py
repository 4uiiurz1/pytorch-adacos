from torchvision import datasets, transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
import numpy as np
import random
from tqdm import tqdm
import os
from PIL import Image
from glob import glob


class Omniglot(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.img_paths[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.img_paths)
