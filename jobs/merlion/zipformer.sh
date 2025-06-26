#!/bin/bash
#$ -N zipformer
#$ -j y -o /export/c09/lavanya/jobs/logs/zinglish/log.zipformer
#$ -M ls1@jh.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c*
#$ -wd /export/c09/lavanya/icefall/egs/multi_zh_en/ASR/
#$ -q g.q

# Clear the log file
> /export/c09/lavanya/jobs/logs/zinglish/log.zipformer

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
directory="/export/c09/lavanya/languageIdentification/zinglish/segmentSmall"

# Get list of all files in the directory
files=$(find "$directory" -type f -name "*.wav")

# Define batch size (adjust according to available memory)
batch_size=1

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
echo "Job ran for $duration seconds." >> /export/c09/lavanya/jobs/logs/zinglish/log.zipformer
