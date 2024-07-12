"""
Training setup

We will use a matching flow model to train the model
We will use the same dataset for all the experiments
It will be imagenet 64x64 (to reduce the computation time)

We use pytorch lightning for the training
"""

#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from schedulefree import AdamWScheduleFree
import lightning.pytorch as pl

from diffusers.models import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

from torchmetrics import MeanSquaredError

from matchingflowexp import dit_models

IMAGE_SIZE = 32

PI = 3.141592653589


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class FlowTrainer(pl.LightningModule):
    """
    Trainer module for the MNIST model.
    """

    def __init__(
        self,
        nb_time_steps=100,
        noise_proba=0.1,
        save_dir="/teamspace/studios/this_studio/",
    ):
        """
        Args:
            hidden_dim (int): hidden dimension of the model
            num_bins (int): number of bins to discretize the data into
            nb_block (int): number of blocks in the model
        """
        super().__init__()

        self.nb_time_steps = nb_time_steps
        self.noise_proba = noise_proba
        self.save_dir = save_dir
        self.nb_channel = 4

        vae_model = "stabilityai/sdxl-vae"
        self.vae = AutoencoderKL.from_pretrained(vae_model)
        self.vae.eval()

        # create the model
        self.model = dit_models.DiT_models["DiT-B/4"](
            input_size=IMAGE_SIZE, in_channels=4
        )

        # self.model = torch.compile(self.model)

        # create the loss function
        self.loss_fn = nn.MSELoss()

        # self.apply(init_weights)
        # metrics to compute the performance in DB
        self.mean_squared_error_val = MeanSquaredError()
        self.mean_squared_error_train = MeanSquaredError()

    def forward(self, data, t, y):
        """
        Forward pass of the model.
        """
        result_logit = self.model(data, t, y)

        return result_logit

    def sampling_fromlogitnormal(
        self,
    ):
        pass

    def compute_loss(self, t, prior, image, labels, train=True):
        """
        Loss compute from raw prior and noise and labels
        """
        gt = t * prior + (1.0 - t) * image
        weight_ponderation = torch.sqrt(1.0 / (1.0 - t) * 2 * 1.0 / (1.0 - t))
        weight_ponderation = weight_ponderation.clamp(0.0, 100.0)

        noise_forecast = self.model(gt, t.squeeze(), labels.long())

        loss = F.mse_loss(
            weight_ponderation * noise_forecast[:, : self.nb_channel, :, :],
            weight_ponderation * prior,
            reduction="mean",
        )

        if train:
            self.mean_squared_error_train(
                weight_ponderation * noise_forecast[:, : self.nb_channel, :, :],
                weight_ponderation * prior,
            )

            return loss
        else:
            self.mean_squared_error_val(
                weight_ponderation * noise_forecast[:, : self.nb_channel, :, :],
                weight_ponderation * prior,
            )

            return loss

    def training_step(self, batch, _):
        """
        Training step.
        """
        # we get the data from the batch
        image, labels = (
            batch["vae_output"].reshape(-1, 4, 32, 32).to(self.device).float(),
            batch["label"],
        )

        labels = [int(class_image) for class_image in labels]
        labels = torch.tensor(labels).to(self.device)


        batch_size = image.shape[0]
        img_w = image.shape[2]

        # now we need to select a random time step between 0 and 1 for all the batch
        t = torch.rand(batch_size, 1, 1, 1).float()
        t = t.to(self.device)

        # TODO later sample from logitnormal distribution
        # we generate the prior dataset (gaussian noise)
        prior = torch.randn(batch_size, self.nb_channel, img_w, img_w).to(self.device)

        loss = self.compute_loss(t, prior, image, labels)

        self.log("loss_training", loss.cpu().detach().numpy().item())

        return loss

    def validation_step(self, batch, _):
        """
        Training step.
        """
        # we get the data from the batch
        image, labels = (
            batch["vae_output"].reshape(-1, 4, 32, 32).to(self.device).float(),
            batch["label"],
        )

        labels = [int(class_image) for class_image in labels]
        labels = torch.tensor(labels).to(self.device)



        batch_size = image.shape[0]
        img_w = image.shape[2]

        # now we need to select a random time step between 0 and 1 for all the batch
        t_level = torch.arange(start=1, end=9, step=1).float() / 10.0  # 8 level
        t = torch.cat([t_level for _ in range(int(batch_size / 8.0))], axis=0).float()
        t = t.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        t = t.to(self.device)

        # TODO later sample from logitnormal distribution
        # we generate the prior dataset (gaussian noise)
        prior = torch.randn(batch_size, self.nb_channel, img_w, img_w).to(self.device)

        loss = self.compute_loss(t, prior, image, labels, train=False)

        return loss

    def on_validation_epoch_end(self):
        """
        We log the metrics at the end of the epoch
        """

        # we log the metrics
        self.log(
            name="val_rmse",
            value=torch.sqrt(self.mean_squared_error_val.compute().detach()),
            sync_dist=True,
        )

        # we reset the metrics
        self.mean_squared_error_val.reset()

    # on training end
    def on_train_epoch_end(self):
        """
        We generate one sample and also
        register the rsme perf
        """
        # we should generate some images
        self.eval()
        with torch.no_grad():
            self.generate()
        self.train()

        # we log the metrics
        self.log(
            name="train_rmse",
            value=torch.sqrt(self.mean_squared_error_train.compute().detach()),
            sync_dist=True,
        )

        # we reset the metrics
        self.mean_squared_error_train.reset()

    def generate(self):
        """
        Method to generate some images.
        """
        # init the prior
        prior_t = torch.randn(1, self.nb_channel, IMAGE_SIZE, IMAGE_SIZE).to(
            self.device
        )

        # choose a random int between 0 and 1000
        y = torch.randint(0, 1000, (1,)).to(self.device)

        for i in range(1, self.nb_time_steps):
            t = torch.ones((1, 1, 1, 1)).to(self.device)
            t = 1.0 - t * i / self.nb_time_steps

            noise_estimation = self.model(
                prior_t, t.squeeze(1).squeeze(1).squeeze(1), y
            )

            u_theta = (
                -1.0 / (1.0 - t) * prior_t
                + 1.0 / (1.0 - t) * noise_estimation[:, : self.nb_channel, :, :]
            )

            prior_t = prior_t - u_theta * 1 / self.nb_time_steps

        image = self.vae.decode(prior_t).sample

        # get the epoch number
        epoch = self.current_epoch
        self.save_image(image, epoch, y.item())

    def save_image(self, data, i, class_attribute):
        """
        Saves the image.
        """
        # plot the data
        # the data has been normalized
        # we clip the data to 0 and 1
        img = VaeImageProcessor().postprocess(
            image=data.detach(), do_denormalize=[True, True]
        )[0]

        name_image = self.save_dir + f"data_{i}_{class_attribute}.png"

        img.save(name_image)

    def configure_optimizers(self):
        """
        Configure the optimizer.
        """
        # create the optimizer
        # optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        optimizer = AdamWScheduleFree(self.model.parameters(), lr=1e-4)

        return optimizer
