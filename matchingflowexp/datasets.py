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

DEFAULT_IMAGE_SIZE = 256

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)


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
            "ILSVRC/imagenet-1k",
            split="train" if self.train else "validation",
            cache_dir=self.root,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index]["image"]
        target = self.dataset[index]["label"]

        if self.transform is not None:
            #try:
            img = self.transform(img)

            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            img = self.vaeprocessor.preprocess(
                img, height=DEFAULT_IMAGE_SIZE, width=DEFAULT_IMAGE_SIZE
            )

            img = img.squeeze(0)

            # except Exception as e:
            #     return self.__getitem__(index + 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
