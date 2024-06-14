"""
Training setup

We will use a matching flow model to train the model
We will use the same dataset for all the experiments
It will be imagenet 64x64 (to reduce the computation time)

We use pytorch lightning for the training
"""

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


import lightning.pytorch as pl

from schedulefree import AdamWScheduleFree

from matchingflowexp import dit_models

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
        noise_proba=0.01,
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

        # create the model
        self.model = dit_models.DiT_models["DiT-S/4"](input_size=64, in_channels=3)

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

    def training_step(self, batch, _):
        """
        Training step.
        """
        # we get the data from the batch
        image, labels = batch

        batch_size = image.shape[0]
        img_w = image.shape[2]
        nb_channel = 3

        # now we need to select a random time step between 0 and 1 for all the batch
        t = torch.rand(batch_size, 1).float()
        t = t.to(self.device)

        w_t, alpha_t, alpha_t_dt = self.compute_params_from_t(t)

        # we generate the prior dataset (gaussian noise)
        prior = torch.randn(batch_size, nb_channel, img_w, img_w).to(self.device)

        alpha_t = alpha_t.unsqueeze(2).unsqueeze(3)

        gt = (1 - alpha_t) * prior + alpha_t * image

        # TODO add some noise to gt
        gt = gt + self.noise_proba * torch.randn(
            batch_size, nb_channel, img_w, img_w
        ).to(self.device)

        result_unnoise = self.model(gt, t.squeeze(1), labels)

        loss = F.mse_loss(result_unnoise[:, :3, :, :], image, reduction="none")

        loss = loss * w_t.unsqueeze(1).unsqueeze(1)
        loss = torch.mean(loss)

        self.log("loss_training", loss.cpu().detach().numpy().item())

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
        prior_t = torch.randn(3, 64, 64, self.num_bins).to(self.device)

        # choose a random class TODO
        y = 0

        for i in range(self.nb_time_steps):
            t = torch.ones((1)).to(self.device)
            t = t * i / self.nb_time_steps

            w_t, alpha_t, alpha_t_dt = self.compute_params_from_t(t)

            g1_estimation = self.model(prior_t, t, y)

            # apply softmax to the logits
            g1_estimation = F.softmax(g1_estimation, dim=1)

            u_theta = w_t.unsqueeze(2).unsqueeze(3) * (
                g1_estimation.permute(0, 2, 3, 1) - prior_t
            )

            prior_t = prior_t + u_theta * 1 / self.nb_time_steps

        self.save_image(prior_t, 100)

    def save_image(self, data, i):
        """
        Saves the image.
        """
        # plot the data
        plt.imshow(data.squeeze().cpu().numpy(), cmap="gray")

        # title
        plt.title(f"data = {i}")

        # save the figure
        plt.savefig(self.save_dir + f"data_{i}.png")

        # close the figure
        plt.close()

    def configure_optimizers(self):
        """
        Configure the optimizer.
        """
        # create the optimizer
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = AdamWScheduleFree(self.parameters(), lr=1e-3)

        return optimizer
