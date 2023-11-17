#!/bin/bash


python main.py \
  --dataset=PACS \
  --save-path=Results/Ours/debug/PACS/resnets18_gem_t_lr1e3\
  --gpu=0 \
  --do-train=True \
  --lr=1e-3 \
  --data-root=../../Datasets/DA \
  --replace \
  --meta-second-order=False \
  --TTA-head=em \
  --model=DomainAdaptor \
  --backbone=resnet18 \
  --batch-size=64 \
  --num-epoch=60 \
  --exp-num=-2 \
  --start-time=0 \
  --times=1 \
  --fc-weight=10.0 \
  --train=deepall \
  --eval=tta_meta \
  --s=1 \
  --loss-names=gem-t \
  --loader=normal


