#!/bin/bash

CUDA_VISIBLE_DEVICES="1" python main.py \
  --dataset='TinyImageNetC' \
  --save-path='pretrained_models/TinyImageNetC-GEM-T/TTA' \
  --load-path='pretrained_models/TinyImageNetC-GEM-T' \
  --gpu=0 \
  --do-train=False \
  --lr=1e-3 \
  --data-root=../data \
  --replace \
  --meta-second-order=False \
  --TTA-head=em \
  --model=DomainAdaptor \
  --backbone=resnet50 \
  --batch-size=128 \
  --num-epoch=60 \
  --exp-num=1 \
  --start-time=0 \
  --times=1 \
  --fc-weight=10.0 \
  --train=deepall \
  --eval=tta_ft \
  --s=1 \
  --loss-names=gem-t \
  --loader=normal
