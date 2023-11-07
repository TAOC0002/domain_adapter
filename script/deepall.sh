#!/bin/bash

python main.py \
  --dataset='OfficeHome' \
  --save-path='pretrained_models/resnet50_OfficeHome' \
  --gpu=0 \
  --do-train=True \
  --lr=1e-3 \
  --data-root = '../../Datasets/DA' \
  --model='erm' \
  --backbone='resnet50' \
  --batch-size=128 \
  --num-epoch=30 \
  \
  --exp-num=-2 \
  --start-time=0 \
  --times=5 \
  --train=deepall \
  --eval=tta_meta \
  --loader='normal' \
  --eval-step=1 \
  --scheduler='step' \
  --lr-decay-gamma=0.1 \
