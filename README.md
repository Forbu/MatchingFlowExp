# MatchingFlowExp
Repo for experiences around matching flow / generative modelling with imagenet

This setup use two things :

- Training with rectified flow setup (aka a better diffusion model) (https://arxiv.org/abs/2209.03003)

- a DiT base architecture (https://github.com/facebookresearch/DiT)

- The training dataset is latent imagenet (https://huggingface.co/datasets/cloneofsimo/imagenet.int8)

### Exemples images

Example of generated images at intermediate training (300k steps) : 

![examples](images/image.png)

This is using the DiT B-4 (medium) model
