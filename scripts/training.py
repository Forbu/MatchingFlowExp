"""
Module to train the model
"""

from matchingflowexp import datasets as ds
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from matchingflowexp.trainer_pl import FlowTrainer

import torch

torch.set_float32_matmul_precision("medium")

CURRENT_DIR = "/home/"

if __name__ == "__main__":
    train_dataset = ds.ImageNet64(
        root= CURRENT_DIR + "data",
        train=True,
        transform=ds.DEFAULT_TRANSFORM,
    )

    batch_size = 128
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    model = FlowTrainer(save_dir=CURRENT_DIR + "results/")

    # compile the model
    # model.compile()

    # wandb logger
    logger = None# .loggers.WandbLogger(project="matchingflowimagenet")

    trainer = pl.Trainer(
        max_time={"hours": 5}, logger=logger, gradient_clip_val=1.0, precision="bf16-mixed"
    )
    trainer.fit(model, train_loader)
