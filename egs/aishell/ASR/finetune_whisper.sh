#!/usr/bin/env bash

export PYTHONPATH=/datas/workspaces/speech/asr/framework/next_gen_kaldi/icefall:$PYTHONPATH

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=1

num_nodes=1
train_engine=torch_ddp

# Automatically detect number and index of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1))) # eg. '0,1,2,3'
else
  num_gpus=-1
  gpu_list="-1"
fi
num_gpus=1
gpu_list="0"
log "num_gpus: ${num_gpus}"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: fine-tuning on Aishell test set on whisper large-v2"
  if [ ${train_engine} == "torch_ddp" ]; then
    log "using torch_ddp"
    # fine-tuning with ddp
    torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
        ./whisper/train.py \
          --max-duration 50 \
          --num-epochs 10 \
          --start-epoch 3 \
          --exp-dir whisper/exp_large_v2 \
          --model-name large-v2
  else
    log "using deepspeed"
    # fine-tuning with deepspeed
    torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
      ./whisper/train.py \
        --max-duration 200 \
        --num-epochs 10 \
        --start-epoch 3 \
        --exp-dir whisper/exp_large_v2 \
        --model-name large-v2 \
        --deepspeed \
        --deepspeed_config ./whisper/ds_config_zero1.json
  fi
fi
