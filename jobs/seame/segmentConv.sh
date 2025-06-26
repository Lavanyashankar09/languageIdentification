#!/bin/bash

#SBATCH --job-name=segmentConv
#SBATCH --output=/export/c09/lavanya/jobs/logs/seame/segment/log.segmentConv
#SBATCH --error=/export/c09/lavanya/jobs/logs/seame/segment/log.segmentConv
#SBATCH --mail-user=ls1@jh.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/export/c09/lavanya/languageIdentification/seameCode
#SBATCH --partition=gpu

source /home/gqin2/scripts/acquire-gpu
echo "cuda device: $CUDA_VISIBLE_DEVICES"
source ~/.bashrc
conda activate capstone
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH="${PATH}:${HOME}/.local/bin"

echo "started running"
python3 2_make_seg_conv.py

