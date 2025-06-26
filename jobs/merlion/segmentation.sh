#!/bin/bash

#$ -N segment
#$ -j y -o /export/c09/lavanya/jobs/logs/zinglish/log.segment
#$ -M ls1@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
#$ -wd /export/c09/lavanya/languageIdentification/zingCode
# Submit to GPU (c0*|c1[0123456789])
#$ -q g.q

source /home/gqin2/scripts/acquire-gpu
echo "cuda device: $CUDA_VISIBLE_DEVICES"
source ~/.bashrc
conda activate capstone
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH="${PATH}:${HOME}/.local/bin"

echo "started running"
python3 2_make_seg.py

