#!/bin/bash

python main.py \
  --dataset=OfficeHome \
  --save-path=AdaBN/meta_OfficeHome \
  --gpu=0 \
  --do-train=True \
  --meta-lr=0.1 \
  --lr=1e-3 \
  --data-root=../../Datasets/DA \
  --replace \
  --meta-step=2 \
  --meta-second-order=False \
  --TTA-head=em \
  --model=DomainAdaptor \
  --backbone=resnet18 \
  --batch-size=64 \
  --num-epoch=30 \
  --exp-num -2 \
  --start-time=0 \
  --times=5 \
  --fc-weight=10.0 \
  --train=tta_meta_sup1 \
  --loader=meta \
  --sup_weight=1 \
  --mix-lambda=0.9 \
  --thresh=0.95 \
  --s=1
