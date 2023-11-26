#!/bin/bash

python main.py \
  --dataset='VISDA' \
  #--save-path='Results/VISDA/resnet18_lr1e-3' \
  --load-path =Results/VISDA/resnet18/resnet18_lr1e-3
  --gpu=0 \
  --do-train=True \
  --lr=1e-3 \
  --data-root ='../../Datasets/DA' \
  --backbone='resnet18' \
  --batch-size=64 \
  --num-epoch=30 \
  --exp-num=-2 \
  --start-time=0 \
  --times=1 \
  --train=deepall \
  --eval=tta_meta \
  --loader='normal' \
  --eval-step=1 \
  --scheduler='step' \
  --lr-decay-gamma=0.1 \
