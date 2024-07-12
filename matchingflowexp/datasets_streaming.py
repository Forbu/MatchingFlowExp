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


from streaming.base.format.mds.encodings import Encoding, _encodings
import numpy as np
from typing import Any
import torch
from streaming import StreamingDataset


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0


def generate_streaming_dataset(
    remote_train_dir="./vae_mds", local_train_dir="./local_train_dir", batch_size=32, split=None
):
    """
    In case of multinode training
    """
    _encodings["uint8"] = uint8

    return StreamingDataset(
        local=local_train_dir,
        remote=remote_train_dir,
        split=split,
        shuffle=True,
        shuffle_algo="naive",
        num_canonical_nodes=1,
        batch_size=batch_size,
    )
