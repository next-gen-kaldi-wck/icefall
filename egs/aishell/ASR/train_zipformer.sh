#!/usr/bin/env bash

export PYTHONPATH=/datas/workspaces/speech/asr/framework/next_gen_kaldi/icefall:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1"

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=1

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1:  train Zipformer BBPE with aishell1"
  python3 ./zipformer/train_bbpe.py \
    --world-size 2 \
    --num-epochs 40 \
    --start-epoch 1 \
    --use-fp16 1 \
    --context-size 2 \
    --exp-dir zipformer/exp_bbpe \
    --max-duration 750 \
    --base-lr 0.045 \
    --lr-batches 7500 \
    --lr-epochs 10 \
#  torchrun --nproc_per_node 1 python -m ipdb ./zipformer/train_bbpe.py \
#    --world-size 2 \
#    --num-epochs 40 \
#    --start-epoch 1 \
#    --use-fp16 1 \
#    --context-size 2 \
#    --exp-dir zipformer/exp_bbpe \
#    --max-duration 1000 \
#    --base-lr 0.045 \
#    --lr-batches 7500 \
#    --lr-epochs 10
fi
