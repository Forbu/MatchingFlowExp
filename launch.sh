cd /home
pip install -r requirements.txt

# uninstall Pillow and install 9.4.0, automatically
pip uninstall Pillow --yes
pip install Pillow==9.4.0

python3 -m scripts.training_v2