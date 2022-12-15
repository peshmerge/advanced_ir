#You can just call this using terminal (eg. ./create_env.sh)
conda create --quiet --yes -n apfir python="3.8.13"
conda install -n apfir ipykernel --yes
conda install -n apfir nb_conda_kernels --yes
conda install -n apfir h5py pandas tqdm Pillow scikit-image scikit-learn scipy --yes
conda update -n base -c defaults conda --yes
conda update --all --yes
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/miniconda3/lib

#After installing conda, do conda activate apfir
#Then, run these commands:
#pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
#python -m ipykernel install --user --name=apfir