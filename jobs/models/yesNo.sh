#!/bin/bash

#$ -N yesnojob
#$ -j y -o /export/c09/lavanya/jobs/logs/log.yesNo
#$ -M ls1@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
#$ -wd /export/c09/lavanya/icefall/egs/yesno/ASR
# Submit to GPU (c0*|c1[0123456789])
#$ -q g.q

source /home/gqin2/scripts/acquire-gpu
echo "cuda device: $CUDA_VISIBLE_DEVICES"
source ~/.bashrc
conda activate capstone
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH="${PATH}:${HOME}/.local/bin"

echo "started running"
 ./prepare.sh
 ./tdnn/train.py
 ./tdnn/decode.py

#./tdnn/pretrained.py --checkpoint ./tmp/icefall_asr_yesno_tdnn/pretrained.pt --words-file ./tmp/icefall_asr_yesno_tdnn/lang_phone/words.txt --HLG ./tmp/icefall_asr_yesno_tdnn/lang_phone/HLG.pt ./tmp/icefall_asr_yesno_tdnn/test_waves/0_0_1_0_1_0_0_1.wav

