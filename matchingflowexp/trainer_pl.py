"""
Training setup

We will use a matching flow model to train the model
We will use the same dataset for all the experiments
It will be imagenet 64x64 (to reduce the computation time)

We use pytorch lightning for the training
"""

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import lightning.pytorch as pl

from diffusers.models import AutoencoderKL
from schedulefree import AdamWScheduleFree

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
        cfg_value=1.2,
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
        self.cfg_value = cfg_value

        vae_model = "stabilityai/sd-vae-ft-ema"
        self.vae = AutoencoderKL.from_pretrained(vae_model)
        #self.vae.eval()

        # create the model
        self.model = dit_models.DiT_models["DiT-B/4"](
            input_size=IMAGE_SIZE, in_channels=4
        )

        self.train()

        # self.model = torch.compile(self.model)

        # create the loss function
        self.loss_fn = nn.MSELoss()

        # self.apply(init_weights)

    def forward(self, data, t, y):
        """
        Forward pass of the model.
        """
        result_logit = self.model(data, t, y)

        return result_logit

    def compute_loss(self, logits, data, init_data):
        """
        Computes the loss.
        """
        return

    def compute_params_from_t(self, t):
        # we generate alpha_t = 1 - cos2(t * pi/2)
        alpha_t = 1 - torch.cos(t * PI / 2) ** 2
        alpha_t_dt = PI * torch.cos(PI / 2 * t) * torch.sin(PI / 2 * t)

        w_t = alpha_t_dt / (1 - alpha_t)

        # make w_t min of 0.005 and max of 1.5
        w_t = torch.clamp(w_t, min=0.005, max=1.5)

        return w_t, alpha_t, alpha_t_dt

    def training_step(self, batch, batch_id):
        """
        Training step.
        """
        # we get the data from the batch
        image, labels = batch

        batch_size = image.shape[0]
        img_w = image.shape[2]

        with torch.no_grad():
            # now we need to select a random time step between 0 and 1 for all the batch
            # to stabilize training we stratify the time generation
            nb_strat = 8.
            t_level = torch.arange(start=0, end=int(nb_strat), step=1).float() * 1./nb_strat  # nb_strat level
            t = torch.cat([t_level + torch.rand(int(nb_strat)) * 1./nb_strat for _ in range(int(batch_size / nb_strat))], axis=0).float()
            t = t.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            t = t.to(self.device)

            weight_ponderation = torch.sqrt(1.0 / (1.0 - t) * 2 * 1.0 / (1.0 - t))
            weight_ponderation = weight_ponderation.clamp(1.0, 5.0)

            # we generate the prior dataset (gaussian noise)
            prior = torch.randn(batch_size, self.nb_channel, img_w, img_w).to(self.device)

            gt = (1 - t) * prior + t * image

        result_noise = self.model(gt, t.squeeze(), labels.long())

        loss = F.mse_loss(
            weight_ponderation * result_noise[:, : self.nb_channel, :, :], weight_ponderation * prior, reduction="mean"
        )

        self.log("loss_training", loss.cpu().detach().numpy().item())

        # print("batch_id : ", batch_id)
        # print("loss_training : ", loss.cpu().detach().numpy().item())

        # if np.isnan(loss.cpu().detach().numpy()).item():
        #     breakpoint()

        return loss

    # on training end
    def on_train_epoch_end(self):
        # we should generate some images
        self.eval()
        with torch.no_grad():
            self.generate()
        self.train()

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
        y_uncond = torch.tensor([1000]).to(self.device)

        for i in range(self.nb_time_steps):
            t = torch.ones((1)).to(self.device)
            t = t * i / self.nb_time_steps

            noise_estimation = self.model(prior_t, t, y)

            u_theta = (
                -1.0 / (1.0 - t) * prior_t
                + 1.0 / (1.0 - t) * noise_estimation[:, : self.nb_channel, :, :]
            )

            noise_estimation_uncond = self.model(prior_t, t, y_uncond)

            u_theta_uncond = (
                -1.0 / (1.0 - t) * prior_t
                + 1.0 / (1.0 - t) * noise_estimation_uncond[:, : self.nb_channel, :, :]
            )

            u_theta_cfg = u_theta_uncond + self.cfg_value * (u_theta - u_theta_uncond)

            prior_t = prior_t - u_theta_cfg * 1. / self.nb_time_steps

        image = self.vae.decode(prior_t / 0.18).sample

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
        data = torch.clamp(data, -1, 1)

        # resize the data to be between 0 and 1
        data = (data + 1.0) / 2.0

        plt.imshow(data.squeeze().cpu().numpy().transpose(1, 2, 0))

        # title
        plt.title(f"data = {class_attribute}")

        # test

        # save the figure
        plt.savefig(self.save_dir + f"data_{i}_{class_attribute}.png")

        # close the figure
        plt.close()

    def configure_optimizers(self):
        """
        Configure the optimizer.
        """
        # create the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        #optimizer = AdamWScheduleFree(self.model.parameters(), lr=1e-4)

        return optimizer
