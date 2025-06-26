#!/bin/bash

#SBATCH --job-name=seameLayer3
#SBATCH --output=/export/c09/lavanya/jobs/logs/seame/trainBiLstm/log.seameLayer3
#SBATCH --error=/export/c09/lavanya/jobs/logs/seame/trainBiLstm/log.seameLayer3
#SBATCH --mail-user=ls1@jh.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/export/c09/lavanya/languageIdentification/seameCode/
#SBATCH --partition=gpu

source /home/gqin2/scripts/acquire-gpu
echo "cuda device: $CUDA_VISIBLE_DEVICES"
source ~/.bashrc
conda activate capstone
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH="${PATH}:${HOME}/.local/bin"

echo "started running"
python 5_train_bilstm.py --hdf5_dir /export/c09/lavanya/languageIdentification/seame/embed/combine/ \
--save_dir_plot /export/c09/lavanya/languageIdentification/seame/plotBiLstm \
--save_dir_cp /export/c09/lavanya/languageIdentification/seame/checkpointBiLstm \
--layer Layer_3 --hidden_dim 128 --batch_size 32 --num_epochs 16 --patience 5 --save_every_n_epochs 4