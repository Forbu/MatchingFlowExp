"""
Module to train the model
"""

from matchingflowexp import datasets as ds
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from matchingflowexp.trainer_pl import FlowTrainer

if __name__ == "__main__":
    train_dataset = ds.ImageNet64(
        root="/teamspace/studios/this_studio/data",
        train=True,
        transform=ds.DEFAULT_TRANSFORM,
    )

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = FlowTrainer()

    # wandb logger
    logger = pl.loggers.WandbLogger(project="normalizeflow")

    trainer = pl.Trainer(max_epochs=20, logger=logger)

    trainer = pl.Trainer(max_epochs=100, logger=logger, accumulate_grad_batches=8)
    trainer.fit(model, train_loader)
