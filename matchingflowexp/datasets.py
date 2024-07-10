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
from diffusers.image_processor import VaeImageProcessor



class ImageNet64(data.Dataset):
    def __init__(
        self, root="./data", train=True, transform=None, target_transform=None
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.vaeprocessor = VaeImageProcessor()

        ## load datasets from huggingface
        self.dataset = datasets.load_dataset(
            "Forbu14/imagenet-1k-latent",
            split="train" if self.train else "validation",
            cache_dir=self.root,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = torch.tensor(self.dataset[index]["latents"]) * 0.18
        target = self.dataset[index]["label_latent"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
