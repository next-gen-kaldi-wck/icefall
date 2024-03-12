#!/usr/bin/env bash

export PYTHONPATH=/datas/workspaces/speech/asr/framework/next_gen_kaldi/icefall:$PYTHONPATH

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=1

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: fine-tuning on Aishell test set on whisper large-v2"
  torchrun --nproc_per_node 8 ./whisper/train.py \
  --max-duration 200 \
  --exp-dir whisper/exp_large_v2 \
  --model-name large-v2 \
  --manifest-dir data/fbank_whisper \
  --deepspeed \
  --deepspeed_config ./whisper/ds_config_zero1.json
fi
