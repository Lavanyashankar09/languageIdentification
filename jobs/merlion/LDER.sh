#!/bin/bash

#SBATCH --job-name=LDLayer3
#SBATCH --output=/export/c09/lavanya/jobs/logs/merlion/trainBiLstm/log.LDLayer3
#SBATCH --error=/export/c09/lavanya/jobs/logs/merlion/trainBiLstm/log.LDLayer3
#SBATCH --mail-user=ls1@jh.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/export/c09/lavanya/languageIdentification/merlionCode
#SBATCH --partition=gpu

source /home/gqin2/scripts/acquire-gpu
echo "cuda device: $CUDA_VISIBLE_DEVICES"
source ~/.bashrc
conda activate capstone
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH="${PATH}:${HOME}/.local/bin"

echo "started running"

python 4_LDER.py --hdf5_dir /export/c09/lavanya/languageIdentification/merlion/embeddingCopy/ \
--save_dir_plot /export/c09/lavanya/languageIdentification/merlion/plotBiLstm/ \
--save_dir_cp /export/c09/lavanya/languageIdentification/merlion/checkpointBiLstm/ \
--layer Layer_3 --hidden_dim 128 --batch_size 32 --num_epochs 1 --patience 5 --save_every_n_epochs 5 \
--checkpoint_path /export/c09/lavanya/languageIdentification/merlion/checkpointBiLstm/Layer_3/model_best.pth