#!/bin/bash
#$ -N zipformer2
#$ -j y -o /export/c09/lavanya/jobs/logs/zinglish/log.zipformer2
#$ -M ls1@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
#$ -wd /export/c09/lavanya/icefall/egs/multi_zh_en/ASR/
# Submit to GPU (c0*|c1[0123456789])
#$ -q g.q

> /export/c09/lavanya/jobs/logs/zinglish/log.zipformer2

source /home/gqin2/scripts/acquire-gpu
echo "cuda device: $CUDA_VISIBLE_DEVICES"
source ~/.bashrc
conda activate capstone
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH="${PATH}:${HOME}/.local/bin"
echo "started running"

./zipformer/pretrained.py \
  --checkpoint ./tmp/icefall-asr-zipformer-multi-zh-en-2023-11-22/exp/pretrained.pt \
  --tokens ./tmp/icefall-asr-zipformer-multi-zh-en-2023-11-22/data/lang_bbpe_2000/tokens.txt \
  --bpe-model ./tmp/icefall-asr-zipformer-multi-zh-en-2023-11-22/data/lang_bbpe_2000/bbpe.model \
  --method modified_beam_search \
 /export/c09/lavanya/languageIdentification/zinglish/small/segmentSmall/TTS_P42566TT_VCST_ECxxx_02_AO_25789991_v001_R007_CRR_MERLIon-CCS_segment_68_Mandarin.wav \
 /export/c09/lavanya/languageIdentification/zinglish/small/segmentSmall/TTS_P42566TT_VCST_ECxxx_02_AO_25789991_v001_R007_CRR_MERLIon-CCS_segment_58_Mandarin.wav