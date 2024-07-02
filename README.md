# MatchingFlowExp
Repo for experiences around matching flow / generative modelling with imagenet

### Interactive mode

```bash
srun -c 4 --gpus=1 --container-remap-root --container-image=pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime --container-name=stardust_prediction_v2 --container-mounts=/opt/marcel-c3/workdir/mccf3797/MatchingFlowExp:/home/ --pty /bin/bash
```