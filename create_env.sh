#!/bin/bash -i

#You can just call this using terminal (eg. ./create_env.sh)

echo "Creating APFIR environment.."
conda create --quiet --yes -n apfir python="3.8.13"
echo "Creating environment done."
echo
echo "Installing required conda packages.."
conda install -n apfir ipykernel --yes
conda install -n apfir nb_conda_kernels --yes
conda install -n apfir h5py pandas tqdm Pillow scikit-image scikit-learn scipy tqdm --yes
conda update -n base -c defaults conda --yes
conda update --all --yes
echo "Installing required conda packages done."
echo
echo "Installing pip packages.."
CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate apfir
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
python3 -m ipykernel install --user --name=apfir
conda deactivate
echo "Installing pip packages done."