cd /home

pip3 install Cython

pip3 install -r requirements.txt

# uninstall Pillow and install 9.4.0, automatically
pip3 uninstall Pillow --yes
pip3 install Pillow==9.4.0

# export HF_HUB_ENABLE_HF_TRANSFER=True
# # actual download script. 
# huggingface-cli download --repo-type dataset cloneofsimo/imagenet.int8 --local-dir ./vae_mds

python3 -m scripts.training_v3