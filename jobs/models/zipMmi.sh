#!/bin/bash
#$ -N zipformerjob
#$ -j y -o /export/c09/lavanya/jobs/logs/log.zipMmi
#$ -M ls1@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
#$ -wd /export/c09/lavanya/icefall/egs/librispeech/ASR/
# Submit to GPU (c0*|c1[0123456789])
#$ -q g.q

# Clear the log file if it exists
> /export/c09/lavanya/jobs/logs/log.zipMmi

source /home/gqin2/scripts/acquire-gpu
echo "cuda device: $CUDA_VISIBLE_DEVICES"
source ~/.bashrc
conda activate capstone
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH="${PATH}:${HOME}/.local/bin"
echo "started running"

./zipformer_mmi/pretrained.py --checkpoint ./tmp/icefall-asr-librispeech-zipformer-mmi-2022-12-08/exp/pretrained.pt --tokens ./tmp/icefall-asr-librispeech-zipformer-mmi-2022-12-08/data/lang_bpe_500/tokens.txt --method 1best ./tmp/icefall-asr-librispeech-zipformer-mmi-2022-12-08/test_wavs/*.wav 


# do this before running
# cd icefall/egs/librispeech/ASR
# mkdir tmp
# cd tmp
# git lfs install 
# git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-mmi-2022-12-08
# [Errno 2] No such file or directory: 'data/lang_bpe_500/tokens.txt'
# put data folder under ASR
#/export/c09/lavanya/icefall/egs/merlion/ASR/download/test/CONFIDENTIAL/audio
# /export/c09/lavanya/merlion_data/output_16khz.wav 