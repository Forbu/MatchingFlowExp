"""
Module to train the model
"""

from matchingflowexp import datasets as ds
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from matchingflowexp.trainer_pl import FlowTrainer

import torch
torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    train_dataset = ds.ImageNet64(
        root="/teamspace/studios/this_studio/data",
        train=True,
        transform=ds.DEFAULT_TRANSFORM,
    )

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model = FlowTrainer()

    # wandb logger
    logger = pl.loggers.WandbLogger(project="matchingflowimagenet")


    trainer = pl.Trainer(max_epochs=100, logger=logger, gradient_clip_val=1.0, precision="bf16")
    trainer.fit(model, train_loader)
