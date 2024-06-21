"""
Module to train the model
"""

import time
import os
import datetime
import sys
import torch

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from matchingflowexp import datasets as ds
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from matchingflowexp.trainer_pl import FlowTrainer


torch.set_float32_matmul_precision("medium")

CURRENT_DIR = "/home/"

DIR_WEIGHTS = CURRENT_DIR + "models/"
NOM_MODELE = "matchingflowv6"
DIR_TB = CURRENT_DIR + "tb_logs/"

# Register callbacks
class MyFairRequeue(Callback):
    """
    Callback to stop the training to requeue the job
    """

    def __init__(self, dir_weights, prefix, max_duration_seconds=7000):
        self.max_duration_seconds = max_duration_seconds
        self.dir = dir_weights
        self.prefix = prefix
        self.start_time = time.monotonic()

    def on_train_batch_end(self, trainer_module, pl_module, _, __, ___):
        """
        Save the model and exit if the training time is too long.
        exit(42) is a magic exit code for slurm to plan requeueing
        """
        if (
            "SLURM_JOB_ID" in os.environ
            and time.monotonic() - self.start_time
            >= self.max_duration_seconds  # avoid requeueing too often
        ):
            print("Time's up, prepare for fair requeueing")

            # we register the date of the current checkpoint
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join(self.dir, self.prefix + "_" + date + ".ckpt")

            trainer_module.save_checkpoint(path)

            path_pth = os.path.join(self.dir, self.prefix + "_" + date + ".pt")
            torch.save(pl_module.state_dict(), path_pth)

            sys.exit(42)  # magic exit code for slurm to plan requeueing


def get_last_checkpoint(dir_weight, nom_model):
    """
    Function to retrieve the last checkpoint.
    """
    # we check if there is a checkpoint in the directory
    list_fichiers = os.listdir(dir_weight)

    # we filter the files to keep only the ones corresponding to the model
    list_fichiers = [f for f in list_fichiers if f.startswith(nom_model)]

    # we sort the files by date of creation
    list_fichiers.sort(key=lambda x: os.path.getmtime(os.path.join(dir_weight, x)))

    # filter : we keep only the files with the extension .ckpt
    list_fichiers = [f for f in list_fichiers if f.endswith(".ckpt")]

    # we get the last file
    if len(list_fichiers) > 0:
        print("checkpoint found")
        return os.path.join(dir_weight, list_fichiers[-1])

    return None


if __name__ == "__main__":
    train_dataset = ds.ImageNet64(
        root=CURRENT_DIR + "data",
        train=True,
    )

    batch_size = 128
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    model = FlowTrainer(save_dir=CURRENT_DIR + "resultsv2/")

    # compile the model
    # model.compile()

    # wandb logger
    logger = None  # .loggers.WandbLogger(project="matchingflowimagenet")
    tb_logger = TensorBoardLogger(DIR_TB, name="matchingflow", version="0.7")

    # get last checkpoint (check the NOM_MODELE and take the last created)
    last_checkpoint = get_last_checkpoint(DIR_WEIGHTS, NOM_MODELE)

    # if there is a checkpoint, load it
    if last_checkpoint is not None:
        model.load_state_dict(torch.load(last_checkpoint)["state_dict"])

        # define the checkpoint callback
    checkpoint_model = MyFairRequeue(
        DIR_WEIGHTS,
        NOM_MODELE,
        max_duration_seconds=7000,
    )

    trainer = pl.Trainer(
        max_time={"hours": 70},
        logger=logger,
        gradient_clip_val=1.0,
        precision="16",
        callbacks=[checkpoint_model],
        #limit_train_batches=0.01,
        enable_progress_bar=True,
        strategy='ddp_find_unused_parameters_true',
    )
    trainer.fit(model, train_loader, ckpt_path=last_checkpoint)
