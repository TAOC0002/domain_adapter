#!/bin/bash

python main.py \
  --dataset='cifar10c' \
  --save-path='pretrained_models/cifar10c/resnet50' \
  --gpu=0 \
  --do-train=True \
  --lr=1e-3 \
  --data-root='../data' \
  --model=DomainAdaptor \
  --backbone='resnet50' \
  --batch-size=512 \
  --num-epoch=20 \
  \
  --exp-num=1 \
  --start-time=0 \
  --train=deepall \
  --eval=deepall \
  --loader='normal' \
  --eval-step=1 \
  --scheduler='step' \
  --lr-decay-gamma=0.1 \
  --corruption fog \
