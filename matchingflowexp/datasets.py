"""
Module for the datasets used in the experiments
We will use the same dataset for all the experiments
It will be imagenet 64x64 (to reduce the computation time)
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import datasets

DEFAULT_IMAGE_SIZE = 64

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ImageNet64(data.Dataset):
    def __init__(self, root="./data", train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        ## load datasets from huggingface
        self.dataset = datasets.load_dataset(
            "ILSVRC/imagenet-1k", split="train" if self.train else "validation", cache_dir=self.root
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        print("begin")
        img = self.dataset[index]["image"]
        target = self.dataset[index]["label"]



        if self.transform is not None:
            img = self.transform(img)

        print(img.shape)

        if img.shape != torch.Size([3, 64, 64]):
            print(img.shape)

        if self.target_transform is not None:
            target = self.target_transform(target)

        

        return img, target