#!/bin/bash

#SBATCH --job-name=zipformerConv-128
#SBATCH --output=/export/c09/lavanya/jobs/logs/seame/zipformer/log.zipformerConv-128
#SBATCH --error=/export/c09/lavanya/jobs/logs/seame/zipformer/log.zipformerConv-128
#SBATCH --mail-user=ls1@jh.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/export/c09/lavanya/icefall/egs/multi_zh_en/ASR/
#SBATCH --partition=gpu

# Start time
start_time=$(date +%s)

source /home/gqin2/scripts/acquire-gpu
echo "cuda device: $CUDA_VISIBLE_DEVICES"
source ~/.bashrc
conda activate capstone
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH="${PATH}:${HOME}/.local/bin"

echo "started running"

# Directory containing the files
directory="/export/c09/lavanya/languageIdentification/seame/segments/conv"

# Get list of all files in the directory
files=$(find "$directory" -type f -name "*.wav")

# Define batch size (adjust according to available memory)
batch_size=128

# Convert files into an array
file_array=($files)

# Process files in batches
for ((i=0; i<${#file_array[@]}; i+=batch_size)); do
    batch="${file_array[@]:i:batch_size}"  # Get batch of files
    echo "Processing batch: ${batch[@]}"
    
    # Run your main command on the current batch
    ./zipformer/pretrained.py \
      --checkpoint ./tmp/icefall-asr-zipformer-multi-zh-en-2023-11-22/exp/pretrained.pt \
      --tokens ./tmp/icefall-asr-zipformer-multi-zh-en-2023-11-22/data/lang_bbpe_2000/tokens.txt \
      --bpe-model ./tmp/icefall-asr-zipformer-multi-zh-en-2023-11-22/data/lang_bbpe_2000/bbpe.model \
      --method modified_beam_search \
      $batch
done

# End time
end_time=$(date +%s)

# Calculate duration
duration=$(( end_time - start_time ))

# Log the duration
echo "Job ran for $duration seconds." >> /export/c09/lavanya/jobs/logs/seame/zipformer/log.zipformerConv-128
