#!/bin/bash

# set exit on error
set -e

echo "1. Install apt packages..."
apt install -y git-lfs
apt install -y libgl1
apt install -y python3-opencv

echo "2. Install python packages..."
pip install -r requirements.txt

echo "3. install LivePortrait, ComfyUI, and download models..."
cd
git clone https://github.com/KwaiVGI/LivePortrait
cd LivePortrait
git lfs install
git clone https://huggingface.co/KwaiVGI/LivePortrait temp_pretrained_weights

mv temp_pretrained_weights/* pretrained_weights/
# comfyui and expression editor
cd 
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
cd custom_nodes
git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait.git
cd ComfyUI-AdvancedLivePortrait 
pip install -r ./requirements.txt

pip install "pydantic<2.0.0" #https://github.com/replicate/cog/issues/1336
# fix onnx cuda runtime lib issue:
pip install  --force-reinstall onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/



echo "4. setup cog server..."
cd
git clone https://github.com/replicate/cog.git
cd cog
cd python
cp ~/crazyface3/python/cog-server/* ./

pip install structlog b2sdk tenacity


echo "5. Kill any running Jupyter Notebook processes..."
ps aux | grep jupyter-notebook | grep -v grep | awk '{print $2}' | xargs -r kill -9

echo "Jupyter Notebook processes have been terminated."

