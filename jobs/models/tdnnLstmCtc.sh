#!/bin/bash
#$ -N conformerjob
#$ -j y -o /export/c09/lavanya/jobs/logs/log.tdnnLstmCtc
#$ -M ls1@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
#$ -wd /export/c09/lavanya/icefall/egs/librispeech/ASR/
# Submit to GPU (c0*|c1[0123456789])
#$ -q g.q

source /home/gqin2/scripts/acquire-gpu

echo "cuda device: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate capstone
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH="${PATH}:${HOME}/.local/bin"

echo "started running"

# do this before running
# cd icefall/egs/librispeech/ASR
# mkdir tmp-lstm 
# cd tmp-lstm 
# git lfs install 
# git clone https://huggingface.co/pkufool/icefall_asr_librispeech_tdnn-lstm_ctc 


python3 ./tdnn_lstm_ctc/pretrained.py \
      --method 1best \
      --checkpoint ./tmp-lstm/icefall_asr_librispeech_tdnn-lstm_ctc/exp/pretrained.pt \
      --words-file ./tmp-lstm/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/words.txt \
      --HLG ./tmp-lstm/icefall_asr_librispeech_tdnn-lstm_ctc/data/lang_phone/HLG.pt \
      ./tmp-lstm/icefall_asr_librispeech_tdnn-lstm_ctc/test_wavs/*.flac